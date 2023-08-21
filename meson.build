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
  version : run_command(files('gen/get_version.py'), 'full', check: true).stdout().strip(),
  license : 'LGPL-3.0-or-later',
  license_files : ['COPYING', 'COPYING.LESSER'],
  default_options : [
    'cpp_std=c++17',
    'b_ndebug=if-release',
    'buildtype=release',
    'warning_level=2',
  ]
)

compiler = meson.get_compiler('cpp')
py = import('python').find_installation()

# Required dependencies
boost_dep = dependency('boost', version : '>=1.69')
dl_dep = dependency('dl')
thread_dep = dependency('threads')
compiler.check_header('libdivide.h', required : true)

# Optional libraries
boost_program_options_dep = dependency('boost', modules : ['program_options'], required : get_option('tools'))
boost_unit_test_framework_dep = dependency('boost', modules : ['unit_test_framework'], required : get_option('unit_test'))
# TODO: fall back to detecting some of these by attempting compilation if pkg-config fails
ibv_dep = dependency('libibverbs', required : get_option('ibv'))
rdmacm_dep = dependency('librdmacm', required : get_option('ibv'))
use_ibv = ibv_dep.found() and rdmacm_dep.found()
if not use_ibv
  # If only one was found, disable them both
  ibv_dep = dependency('', required : false)
  rdmacm_dep = dependency('', required : false)
endif
mlx5_dep = dependency('libmlx5', required : get_option('mlx5dv'))
pcap_dep = dependency('libpcap', required : get_option('pcap'))
cap_dep = dependency('libcap', required : get_option('cap'))
cuda_dep = dependency('cuda', required : get_option('cuda'))
# TODO: detection for gdrapi

# Optional features
use_ibv_hw_rate_limit = get_option('ibv_hw_rate_limit').require(
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
conf.set('SPEAD2_MAJOR', run_command(get_version, 'major', check : true).stdout().strip())
conf.set('SPEAD2_MINOR', run_command(get_version, 'minor', check : true).stdout().strip())
conf.set('SPEAD2_PATCH', run_command(get_version, 'patch', check : true).stdout().strip())
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

gen_loader = files('gen/gen_loader.py')
gen_templates = files('gen/template.cpp', 'gen/template.h')
loader_modules = ['rdmacm', 'ibv', 'mlx5dv']
subdir('include/spead2')  # Generates include/spead2/common_features.h and common_loader_*.h

include_dir = include_directories('include')
subdir('src')  # Defines the targets

pkg = import('pkgconfig')
pkg.generate(
  st_lib,
  description : 'Library for sending and receiving data in the SPEAD protocol',
  filebase : 'spead2',
)
