#!/usr/bin/env python

# Copyright 2019-2020 National Research Foundation (SARAO)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import copy
import os
import sys

import jinja2
from pycparser import c_ast
from pycparser.c_parser import CParser
from pycparser.c_generator import CGenerator


# The typedefs are arbitrary and just used to allow pycparser to parse the
# code. Note that some functions in infiniband/verbs.h are implemented as
# static inline functions, and so do not get listed here.
IBV_DECLS = '''
typedef unsigned long size_t;
typedef unsigned long uint64_t;

void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int nevents);

struct ibv_pd *ibv_alloc_pd(struct ibv_context *context);

int ibv_close_device(struct ibv_context *context);

struct ibv_comp_channel *ibv_create_comp_channel(struct ibv_context *context);

struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe,
                             void *cq_context,
                             struct ibv_comp_channel *channel,
                             int comp_vector);

struct ibv_qp *ibv_create_qp(struct ibv_pd *pd,
                             struct ibv_qp_init_attr *qp_init_attr);

int ibv_dealloc_pd(struct ibv_pd *pd);

int ibv_dereg_mr(struct ibv_mr *mr);

int ibv_destroy_comp_channel(struct ibv_comp_channel *channel);

int ibv_destroy_cq(struct ibv_cq *cq);

int ibv_destroy_qp(struct ibv_qp *qp);

void ibv_free_device_list(struct ibv_device **list);

int ibv_get_cq_event(struct ibv_comp_channel *channel,
                     struct ibv_cq **cq, void **cq_context);

uint64_t ibv_get_device_guid(struct ibv_device *device);

struct ibv_device **ibv_get_device_list(int *num_devices);

struct ibv_context *ibv_open_device(struct ibv_device *device);

int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                  int attr_mask);

int ibv_query_device(struct ibv_context *context,
                     struct ibv_device_attr *device_attr);

struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr,
                          size_t length, int access);

struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *addr, size_t length,
                                uint64_t iova, unsigned int access);
'''

RDMACM_DECLS = '''
int rdma_bind_addr(struct rdma_cm_id *id, struct sockaddr *addr);

struct rdma_event_channel *rdma_create_event_channel(void);

int rdma_create_id(struct rdma_event_channel *channel,
                   struct rdma_cm_id **id, void *context,
                   enum rdma_port_space ps);

void rdma_destroy_event_channel(struct rdma_event_channel *channel);

int rdma_destroy_id(struct rdma_cm_id *id);
'''

MLX5DV_DECLS = '''
typedef int bool;
typedef unsigned long uint64_t;

bool mlx5dv_is_supported(struct ibv_device *device);

int mlx5dv_query_device(struct ibv_context *ctx_in,
                        struct mlx5dv_context *attrs_out);

struct ibv_wq *mlx5dv_create_wq(struct ibv_context *context,
                                struct ibv_wq_init_attr *wq_init_attr,
                                struct mlx5dv_wq_init_attr *mlx5_wq_attr);

int mlx5dv_init_obj(struct mlx5dv_obj *obj, uint64_t obj_type);
'''


class RenameVisitor(c_ast.NodeVisitor):
    """Renames a function in the AST.

    The function name is stored both in the Decl and in the inner-most TypeDecl
    of the return type. This only handles the latter.
    """

    def __init__(self, old_name, new_name):
        super(RenameVisitor, self).__init__()
        self.old_name = old_name
        self.new_name = new_name

    def visit_TypeDecl(self, node):
        if node.declname == self.old_name:
            node.declname = self.new_name


def rename_func(func, new_name):
    """Return a copy of a function declaration with a new name"""
    func = copy.deepcopy(func)
    RenameVisitor(func.name, new_name).visit(func)
    func.name = new_name
    return func


def make_func_ptr(func, with_name=True):
    """Create a node of pointer-to-function type."""
    if not with_name:
        node = rename_func(func, None)
    else:
        node = copy.deepcopy(func)
    node.type = c_ast.PtrDecl(quals=[], type=node.type)
    return node


def func_args(func):
    """Get list of function argument names"""
    args = [arg.name for arg in func.type.args.params]
    # Handle (void)
    if args == [None]:
        args = []
    return args


class Visitor(c_ast.NodeVisitor):
    """Collects all the function definitions."""

    def __init__(self):
        self.nodes = []

    def visit_Decl(self, node):
        if not isinstance(node.type, c_ast.FuncDecl):
            return
        self.nodes.append(node)


def gen_node(node):
    return CGenerator().visit(node)


class Library:
    """Symbols from a single DSO.

    Parameters
    ----------
    name : str
        Short name used to form symbol names.
    headers : Iterable[str]
        Header files to include.
    soname : str
        Name of library to load.
    guard : str
        Preprocessor expression governing conditional compilation.
    decls : str
        Prototypes of functions to be wrapped.
    wrappers : Iterable[str]
        Functions for which an implementation will be provided in the global
        namespace.
    optional : Iterable[str]
        Functions whose absence doesn't prevent the library being loaded. Calling
        such a function raises a std::system_error. The presence can be tested with
        another function, formed by prefixing the original function name with
        @c has_.
    """

    def __init__(self, name, headers, soname, guard, decls, wrappers=(), optional=(),
                 *, fail_log_level='warning'):
        self.name = name
        self.headers = list(headers)
        self.soname = soname
        self.guard = guard
        ast = CParser().parse(decls)
        visitor = Visitor()
        visitor.visit(ast)
        self.nodes = visitor.nodes
        self.wrappers = list(wrappers)
        self.environment = jinja2.Environment(
            autoescape=False,
            trim_blocks=True,
            loader=jinja2.FileSystemLoader(os.path.dirname(__file__))
        )
        self.environment.filters['gen'] = gen_node
        self.environment.filters['rename'] = rename_func
        self.environment.filters['ptr'] = make_func_ptr
        self.environment.filters['args'] = func_args
        self.environment.globals['name'] = self.name
        self.environment.globals['headers'] = self.headers
        self.environment.globals['soname'] = self.soname
        self.environment.globals['guard'] = self.guard
        self.environment.globals['nodes'] = self.nodes
        self.environment.globals['wrappers'] = set(wrappers)
        self.environment.globals['optional'] = set(optional)
        self.environment.globals['fail_log_level'] = fail_log_level

    def header(self):
        return self.environment.get_template('template.h').render()

    def cxx(self):
        return self.environment.get_template('template.cpp').render()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['header', 'cxx'])
    parser.add_argument('library', choices=['rdmacm', 'ibv', 'mlx5dv'])
    args = parser.parse_args()

    if args.library == 'rdmacm':
        lib = Library('rdmacm', ['rdma/rdma_cma.h'], 'librdmacm.so.1', 'SPEAD2_USE_IBV',
                      RDMACM_DECLS)
    elif args.library == 'ibv':
        lib = Library('ibv', ['infiniband/verbs.h'], 'libibverbs.so.1', 'SPEAD2_USE_IBV',
                      IBV_DECLS,
                      ['ibv_create_qp', 'ibv_query_device'],
                      ['ibv_reg_mr_iova2'])
    else:
        lib = Library('mlx5dv', ['infiniband/mlx5dv.h'], 'libmlx5.so.1', 'SPEAD2_USE_MLX5DV',
                      MLX5DV_DECLS, fail_log_level='debug')

    if args.type == 'header':
        print(lib.header())
    else:
        print(lib.cxx())


if __name__ == '__main__':
    main(sys.argv)
