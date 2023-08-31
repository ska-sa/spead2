# Copyright 2023 National Research Foundation (SARAO)
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

project(
  'spead2',
  'cpp',
  meson_version : '>=1.2.0',
  version : files('VERSION.txt'),
  license : 'LGPL-3.0-or-later',
  license_files : ['COPYING', 'COPYING.LESSER'],
  default_options : [
    'cpp_std=c++17',
    'cuda_std=c++17',
    'b_ndebug=if-release',
    'buildtype=release',
    'default_library=static',
    'warning_level=2',
    'b_staticpic=false',  # The default of true harms performance
  ]
)
shared_lib_version = '5.0.0'

compiler = meson.get_compiler('cpp')
py = import('python').find_installation(pure : false, modules : ['jinja2', 'pycparser', 'packaging'])

# Required dependencies
boost_dep = dependency('boost', version : '>=1.69')
dl_dep = dependency('dl')
thread_dep = dependency('threads')
compiler.check_header('libdivide.h', required : true)

# Optional libraries
boost_program_options_dep = dependency(
  'boost',
  modules : ['program_options'],
  required : get_option('tools'),
  disabler : true
)
boost_unit_test_framework_dep = dependency(
  'boost',
  modules : ['unit_test_framework'],
  required : get_option('unit_test'),
  disabler : true
)
if add_languages('cuda', required : get_option('cuda'), native : false)
  # Empty but found dependency; the compiler takes care of CUDA so we don't
  # need to explicitly link to anything.
  cuda_dep = declare_dependency()
else
  cuda_dep = disabler()
endif

# Meson doesn't support has_function for cuda, so we have to check manually
gdrapi_dep = disabler()  # Assume not found until proven otherwise
gdrapi_opt = get_option('gdrapi').require(cuda_dep.found())
gdrapi_test_dep = declare_dependency(link_args : '-lgdrapi')
if gdrapi_opt.allowed() and meson.get_compiler('cuda').links(
  '''
  #include <gdrapi.h>
  int main() { (void) gdr_open(); }
  ''',
  name : 'gdrapi library',
  # Workaround for https://github.com/mesonbuild/meson/issues/12169
  args : get_option('cuda_args'),
  dependencies : [gdrapi_test_dep],
  no_builtin_args : false,
)
  gdrapi_dep = gdrapi_test_dep
endif
gdrapi_opt.require(gdrapi_dep.found())

# The rdma-core libraries have pkgconfig files in newer versions, but older
# versions did not and we still want to support those. This unfortunately
# requires more complex logic.
libraries = [
  {
    'name': 'ibv',
    'option': 'ibv',
    'dep_name': 'libibverbs',
    'header': 'infiniband/verbs.h',
    'lib': 'ibverbs',
    'function': 'ibv_create_wq',
  },
  {
    'name': 'rdmacm',
    'option': 'ibv',
    'dep_name': 'librdmacm',
    'header': 'rdma/rdma_cma.h',
    'lib': 'rdmacm',
    'function': 'rdma_create_event_channel',
  },
  {
    'name': 'mlx5',
    'option': 'mlx5dv',
    'dep_name': 'libmlx5',
    'header': 'infiniband/mlx5dv.h',
    'lib': 'mlx5',
    'function': 'mlx5dv_create_cq',
    'prereq': ['ibv_dep', 'rdmacm_dep'],
  },
]
foreach lib : libraries
  opt = get_option(lib['option'])
  foreach prereq : lib.get('prereq', [])
    opt = opt.require(get_variable(prereq).found())
  endforeach
  if opt.allowed()
    lib_dep = dependency(lib['dep_name'], required : false)
    if not lib_dep.found() and compiler.has_function(
      lib['function'],
      args : '-l' + lib['lib'],
      prefix : '#include <' + lib['header'] + '>',
    )
      lib_dep = declare_dependency(link_args : '-l' + lib['lib'])
    endif
  else
    lib_dep = dependency('', required : false)  # A not-found dependency
  endif
  opt.require(lib_dep.found())
  set_variable(lib['name'] + '_dep', lib_dep)
endforeach
use_ibv = ibv_dep.found() and rdmacm_dep.found()

# Note: pass 'pcap' not 'libpcap' to dependency, to trigger the special
# handling that uses pcap-config if pkgconfig doesn't find it.
pcap_dep = dependency('pcap', required : get_option('pcap'))
cap_dep = dependency('libcap', required : get_option('cap'), disabler : true)

# Optional features
use_ibv_hw_rate_limit = use_ibv and get_option('ibv_hw_rate_limit').require(
  compiler.has_function(
    'ibv_modify_qp_rate_limit',
    dependencies : [ibv_dep, rdmacm_dep],
    prefix : '#include <infiniband/verbs.h>'
  )
).allowed()
use_recvmmsg = get_option('recvmmsg').require(
  compiler.has_function(
    'recvmmsg',
    args : '-D_GNU_SOURCE',
    prefix : '#include <sys/socket.h>'
  )
).allowed()
use_sendmmsg = get_option('sendmmsg').require(
  compiler.has_function(
    'sendmmsg',
    args : '-D_GNU_SOURCE',
    prefix : '#include <sys/socket.h>'
  )
).allowed()
use_eventfd = get_option('eventfd').require(
  compiler.has_function(
    'eventfd',
    prefix : '#include <sys/eventfd.h>'
  )
).allowed()
use_posix_semaphores = get_option('posix_semaphores').require(
  compiler.has_function(
    'sem_init',
    prefix : '''
      #include <unistd.h>
      #include <semaphore.h>

      #if defined(__APPLE__)
      # error "POSIX semaphores do not work on OS X"
      #endif
      #if !defined(_POSIX_SEMAPHORES) || _POSIX_SEMAPHORES < 0
      # error "POSIX semaphores marked unsupported"
      #endif
    ''',
    dependencies : thread_dep
  )
).allowed()
use_pthread_setaffinity_np = get_option('pthread_setaffinity_np').require(
  compiler.has_function(
    'pthread_setaffinity_np',
    args : '-D_GNU_SOURCE',
    prefix : '#include <pthread.h>',
    dependencies : thread_dep
  )
).allowed()
use_fmv = get_option('fmv').require(
  compiler.compiles(
    '__attribute__((target("default"))) void foo() {}',
    args : '-Werror',  # TODO: use werror kwarg once meson 1.3 is in use
    name : 'function multi-versioning'
  )
).allowed()
# has_function doesn't work for _mm_stream_si128, because it is an
# inline-only function in GCC without an external definition.
use_movntdq = get_option('movntdq').require(
  compiler.compiles(
    '''
    #include <emmintrin.h>

    void foo()
    {
        _mm_stream_si128((__m128i *) NULL, __m128i());
    }
    ''',
    name : 'MOVNTDQ intrinsic'
  )
).allowed()

# Write configuration data
conf = configuration_data()
conf.set('SPEAD2_VERSION', meson.project_version())
get_version = files('gen/get_version.py')
conf.set('SPEAD2_MAJOR', run_command(py, get_version, 'major', meson.project_version(), check : true).stdout().strip())
conf.set('SPEAD2_MINOR', run_command(py, get_version, 'minor', meson.project_version(), check : true).stdout().strip())
conf.set('SPEAD2_PATCH', run_command(py, get_version, 'patch', meson.project_version(), check : true).stdout().strip())
conf.set10('SPEAD2_USE_IBV', use_ibv)
conf.set10('SPEAD2_USE_IBV_HW_RATE_LIMIT', use_ibv_hw_rate_limit)
conf.set10('SPEAD2_USE_MLX5DV', mlx5_dep.found())
conf.set10('SPEAD2_USE_RECVMMSG', use_recvmmsg)
conf.set10('SPEAD2_USE_SENDMMSG', use_sendmmsg)
conf.set10('SPEAD2_USE_EVENTFD', use_eventfd)
conf.set10('SPEAD2_USE_POSIX_SEMAPHORES', use_posix_semaphores)
conf.set10('SPEAD2_USE_PTHREAD_SETAFFINITY_NP', use_pthread_setaffinity_np)
conf.set10('SPEAD2_USE_FMV', use_fmv)
conf.set10('SPEAD2_USE_MOVNTDQ', use_movntdq)
conf.set10('SPEAD2_USE_PCAP', pcap_dep.found())
conf.set('SPEAD2_MAX_LOG_LEVEL', '(spead2::log_level::' + get_option('max_log_level') + ')')

gen_loader = files('gen/gen_loader.py')
gen_templates = files('gen/template.cpp', 'gen/template.h')
loader_modules = ['rdmacm', 'ibv', 'mlx5dv']
subdir('include/spead2')  # Generates include/spead2/common_features.h and common_loader_*.h

include_dir = include_directories('include')
subdir('src')  # Defines the targets

if not get_option('python')
  subdir('examples')

  pkg = import('pkgconfig')
  pkg.generate(
    st_lib,
    description : 'Library for sending and receiving data in the SPEAD protocol',
    filebase : 'spead2',
  )
endif
