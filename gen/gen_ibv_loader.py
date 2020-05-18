#!/usr/bin/env python

# Copyright 2019 SKA South Africa
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

import sys
import copy

import jinja2
from pycparser import c_ast
from pycparser.c_parser import CParser
from pycparser.c_generator import CGenerator


PREFIX = '''
/* Copyright 2019 SKA South Africa
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 *
 * This file is automatically generated. Do not edit.
 */
'''

# Extracted signatures for symbols needed from libibverbs. The typedefs are
# arbitrary and just used to allow pycparser to parse the code.
#
# Note that some functions in infiniband/verbs.h are implemented as static
# inline functions, and so do not get listed here.
INPUT = '''
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


int rdma_bind_addr(struct rdma_cm_id *id, struct sockaddr *addr);

struct rdma_event_channel *rdma_create_event_channel(void);

int rdma_create_id(struct rdma_event_channel *channel,
                   struct rdma_cm_id **id, void *context,
                   enum rdma_port_space ps);

void rdma_destroy_event_channel(struct rdma_event_channel *channel);

int rdma_destroy_id(struct rdma_cm_id *id);

'''

HEADER = PREFIX + '''\
#ifndef SPEAD2_COMMON_IBV_LOADER_H
#define SPEAD2_COMMON_IBV_LOADER_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>

#if SPEAD2_USE_IBV

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

{% for node in nodes -%}
#undef {{ node.name }}
{% endfor %}

namespace spead2
{

{% for node in nodes -%}
extern {{ node | ptr | gen }};
{% endfor %}

// Load ibverbs. If it could not be loaded, throws std::system_error
void ibv_loader_init();

} // namespace spead2

#endif // SPEAD2_USE_IBV
#endif // SPEAD2_COMMON_IBV_LOADER_H
'''

CXX = '''\
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>

#if SPEAD2_USE_IBV

#include <spead2/common_ibv_loader.h>
#include <spead2/common_ibv_loader_utils.h>
#include <spead2/common_logging.h>
#include <mutex>
#include <exception>

namespace spead2
{

static std::once_flag init_once;
static std::exception_ptr init_result;

{% for node in nodes %}
{{ node | rename(node.name + '_stub') | gen }}
{
{% for arg in (node | args) %}
    (void) {{ arg }};
{% endfor %}
    ibv_loader_stub(init_result);
}
{% endfor %}

{% for node in nodes %}
{{ node | ptr | gen }} = {{ node.name }}_stub;
{% endfor %}

static void reset_stubs()
{
{% for node in nodes %}
    {{ node.name }} = {{ node.name }}_stub;
{% endfor %}
}

static void init()
{
    try
    {
        dl_handle librdmacm("librdmacm.so.1");
        dl_handle libibverbs("libibverbs.so.1");
{% for node in nodes %}
{% set lib = 'libibverbs' if node.name.startswith('ibv_') else 'librdmacm' %}
        {{ node.name }} = reinterpret_cast<{{ node | ptr(False) | gen }}>(
            {{ lib }}.sym("{{ node.name }}"));
{% endfor %}
        // Prevent the libraries being closed, so that the symbols stay valid
        librdmacm.release();
        libibverbs.release();
    }
    catch (std::system_error &e)
    {
        init_result = std::current_exception();
        reset_stubs();
        log_warning("could not load ibverbs: %s", e.what());
    }
}

void ibv_loader_init()
{
    std::call_once(init_once, init);
    if (init_result)
        std::rethrow_exception(init_result);
}

} // namespace spead2

/* Wrappers in the global namespace. This is needed because ibv_exp_create_qp
 * calls ibv_create_qp, and so we need to provide an implementation.
 */
{% for node in nodes if node.name in ['ibv_create_qp'] %}
{{ node | gen }}
{
    return spead2::{{ node.name }}({{ node | args | join(', ') }});
}
{% endfor %}

#endif // SPEAD2_USE_IBV
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
    """Collects all the function definitions"""
    def __init__(self):
        self.nodes = []

    def visit_Decl(self, node):
        if not isinstance(node.type, c_ast.FuncDecl):
            return
        self.nodes.append(node)


def gen_node(node):
    return CGenerator().visit(node)


def main(argv):
    environment = jinja2.Environment(autoescape=False, trim_blocks=True)
    environment.filters['gen'] = gen_node
    environment.filters['rename'] = rename_func
    environment.filters['ptr'] = make_func_ptr
    environment.filters['args'] = func_args
    header = environment.from_string(HEADER)
    cxx = environment.from_string(CXX)

    ast = CParser().parse(INPUT)
    visitor = Visitor()
    visitor.visit(ast)
    header_text = header.render(nodes=visitor.nodes)
    cxx_text = cxx.render(nodes=visitor.nodes)
    if len(argv) != 2 or argv[1] not in {'header', 'cxx'}:
        print('Usage: {} header|cxx'.format(argv[0], file=sys.stderr))
        return 1
    elif argv[1] == 'header':
        print(header_text)
    else:
        print(cxx_text)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
