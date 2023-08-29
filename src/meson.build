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

# Generated sources
gen_sources = []
foreach name : loader_modules
  cpp_name = f'common_loader_@name@.cpp'
  gen_cpp = custom_target(
    input : [gen_loader],
    output : [cpp_name],
    depend_files : gen_templates,
    command : [py, '@INPUT@', 'cxx', name, '@OUTPUT@']
  )
  gen_sources += gen_cpp
endforeach

# Common sources
ssmod = import('sourceset')
ss = ssmod.source_set()
ss.add(
  files(
    'common_flavour.cpp',
    'common_loader_utils.cpp',
    'common_ibv.cpp',
    'common_inproc.cpp',
    'common_logging.cpp',
    'common_memcpy.cpp',
    'common_memory_allocator.cpp',
    'common_memory_pool.cpp',
    'common_raw_packet.cpp',
    'common_semaphore.cpp',
    'common_socket.cpp',
    'common_thread_pool.cpp',
    'recv_chunk_stream.cpp',
    'recv_chunk_stream_group.cpp',
    'recv_heap.cpp',
    'recv_inproc.cpp',
    'recv_live_heap.cpp',
    'recv_mem.cpp',
    'recv_packet.cpp',
    'recv_ring_stream.cpp',
    'recv_stream.cpp',
    'recv_tcp.cpp',
    'recv_udp_base.cpp',
    'recv_udp.cpp',
    'recv_udp_ibv.cpp',
    'recv_udp_ibv_mprq.cpp',
    'recv_udp_pcap.cpp',
    'send_heap.cpp',
    'send_inproc.cpp',
    'send_packet.cpp',
    'send_streambuf.cpp',
    'send_stream.cpp',
    'send_stream_config.cpp',
    'send_tcp.cpp',
    'send_udp.cpp',
    'send_udp_ibv.cpp',
    'send_writer.cpp',
  )
)
ss.add(gen_sources, gen_headers)
ss.add(
  boost_dep,
  ibv_dep,
  rdmacm_dep,
  mlx5_dep,
  pcap_dep,
  dl_dep,
  thread_dep,
)
ssconfig = ss.apply(conf)

# Targets

if not get_option('python')
  st_lib = library(
    'spead2',
    include_directories: include_dir,
    sources: ssconfig.sources(),
    dependencies: ssconfig.dependencies(),
    install : true,
    version : shared_lib_version
  )

  if get_option('default_library') == 'both'
    # Link command-line tools against the static library, even if the
    # shared library is built.
    st_lib = st_lib.get_static_lib()
  endif

  st_dep = declare_dependency(
    include_directories : include_dir,
    sources : gen_headers,
    link_with : st_lib
  )

  spead2_recv = executable(
    'spead2_recv',
    'spead2_recv.cpp', 'spead2_cmdline.cpp',
    dependencies : [boost_program_options_dep, st_dep],
    install : true
  )
  # Can reuse cmdline.o; no need to bother making a static lib for it
  cmdline_o = spead2_recv.extract_objects('spead2_cmdline.cpp')
  executable(
    'spead2_send',
    'spead2_send.cpp',
    objects : [cmdline_o],
    dependencies : [boost_program_options_dep, st_dep],
    install : true
  )
  executable(
    'spead2_bench',
    'spead2_bench.cpp',
    objects : [cmdline_o],
    dependencies : [boost_program_options_dep, st_dep],
    install : true
  )
  if use_ibv
    executable(
      'mcdump',
      'mcdump.cpp',
      dependencies: [boost_program_options_dep, st_dep],
      install : true
    )
  endif
  executable('spead2_net_raw', 'spead2_net_raw.cpp', dependencies : [cap_dep], install : true)

  unit_test = executable(
    'spead2_unit_test',
    'unittest_main.cpp',
    'unittest_logging.cpp',
    'unittest_memcpy.cpp',
    'unittest_memory_allocator.cpp',
    'unittest_memory_pool.cpp',
    'unittest_raw_packet.cpp',
    'unittest_recv_live_heap.cpp',
    'unittest_recv_custom_memcpy.cpp',
    'unittest_recv_stream_stats.cpp',
    'unittest_semaphore.cpp',
    'unittest_send_heap.cpp',
    'unittest_send_streambuf.cpp',
    'unittest_send_tcp.cpp',
    cpp_args : '-DBOOST_TEST_DYN_LINK',
    dependencies : [st_dep, boost_unit_test_framework_dep]
  )
  test('unit tests', unit_test)

else  # python

  py_sources = files(
    'py_common.cpp',
    'py_recv.cpp',
    'py_register.cpp',
    'py_send.cpp',
  )
  # While meson provides special support for dependency('pybind11'), it's
  # not guaranteed to pick up the version installed in the isolated build
  # environment e.g. if there is a different version on PKG_CONFIG_PATH.
  pybind11_inc = run_command(
    py, '-c', 'import pybind11; print(pybind11.get_include())',
    check : true
  ).stdout().strip()
  pybind11_dep = declare_dependency(include_directories : pybind11_inc)
  # We don't need to pass boost objects across shared library
  # boundaries. These macros makes -fvisibility=hidden do its job.
  # The first is asio-specific, while the latter is only used in
  # Boost 1.81+.
  py_cpp_args = [
    '-DBOOST_ASIO_DISABLE_VISIBILITY',
    '-DBOOST_DISABLE_EXPLICIT_SYMBOL_VISIBILITY',
  ]
  split_debug = get_option('python_split_debug')
  if split_debug
    py_cpp_args += '-g'
  endif
  py_install_dir = py.get_install_dir() / 'spead2'
  py_ext = py.extension_module(
    split_debug ? '_spead2_tmp' : '_spead2',
    [ssconfig.sources(), py_sources],
    cpp_args : py_cpp_args,
    gnu_symbol_visibility : 'inlineshidden',
    include_directories : include_dir,
    dependencies : [pybind11_dep, ssconfig.dependencies()],
    install : not split_debug,
    install_tag : 'python-runtime',
    subdir : 'spead2'
  )
  if split_debug
    fs = import('fs')
    target = fs.name(py_ext.full_path()).replace('_spead2_tmp', '_spead2')
    custom_target(
      input : [py_ext],
      output : [target, target + '.debug'],
      command : [py, files('../gen/strip_debug.py'), '@INPUT@', '@OUTPUT0@', '@OUTPUT1@'],
      install : true,
      install_tag : ['python-runtime', 'devel'],
      install_dir : [py_install_dir, py_install_dir / '.debug']
    )
  endif

  subdir('spead2')
endif
