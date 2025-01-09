Changelog
=========

.. rubric:: 4.4.0

- Add wheels for Python 3.13.
- Drop support for Python 3.8, which has reached end-of-life.
- Update wheels to manylinux_2_28.
- Bump minimum Boost version to 1.70 (this was the practical lower limit,
  as it wasn't compiling against older versions).
- Support Boost 1.87.
- Add non-temporal memcpy support on AArch64 (with SVE).
- Rename :cpp:class:`!spead2::io_service_ref` to
  :cpp:class:`spead2::io_context_ref` to reflect name changes in Asio (the old
  name is retained as a typedef).
- Rename member functions called :func:`!get_io_service` to
  :func:`!get_io_context`, again to reflect name changes in Boost.
- Update URL to download Boost in cibuildwheel configuration.
- Update cibuildwheel and other build and test dependencies.
- Update some unit tests to use :c:macro:`!BOOST_TEST`.
- Fix building documentation as PDF.

.. rubric:: 4.3.2

- Speed up receiving UDP with the Linux kernel network stack by using
  generic receive offload (GRO).
- Update Boost version in wheels to 1.85.
- Fix compatibility with numpy 2.0.
- Change the unit tests to allocate TCP and UDP ports dynamically.
- Add a series of tutorials to the manual.
- Add a missing type annotation for :py:meth:`spead2.ThreadPool.set_affinity`.
- Add a Jenkins configuration to test the ibverbs support on a SARAO-internal
  Jenkins server.
- Bump versions of dependencies used in CI.

.. rubric:: 4.3.1

- Switch from netifaces to netifaces2 for testing.
- Update coverallsapp/github-action to v2.2.3.
- Fix the type annotation for :py:class:`spead2.recv.StreamConfig` to allow
  `explicit_start` to be passed as a constructor argument.

.. rubric:: 4.3.0

- Add ability to override the transmission rate for individual heaps.
- Fix missing type annotation on the `substream_index` argument to
  :py:meth:`~.SyncStream.send_heap`.

.. rubric:: 4.2.0

- Significantly speed up transmission when using the Linux kernel networking
  stack, by using generic segmentation offload.
- Speed up transmission for small packets (20% in some cases).
- Make :cpp:func:`~spead2::send::stream::async_send_heap` and
  :cpp:func:`~spead2::send::stream::async_send_heaps` accept completion
  tokens. This allows for code like
  ``stream.async_send_heap(heap, boost::asio::use_future);``
- Add C++ support to iterate over :cpp:class:`~spead2::recv::ring_stream` and
  :cpp:class:`~spead2::ringbuffer` using a range-based ``for`` loop.
- Add support for the `prefer_huge` argument to the :py:class:`~.MmapAllocator`
  constructor in Python.
- Make the `allocator` argument to :py:class:`~.MemoryPool` optional in
  Python.
- Allow indexing a :py:class:`~.HeapReferenceList` with a slice to create a
  new :py:class:`~.HeapReferenceList` with a subset of the heaps.
- Add :cpp:class:`~spead2::ringbuffer` to the C++ documentation.
- Eliminate use of the deprecated :cpp:var:`!boost::asio::null_buffers`.
- Update Boost version in wheels to 1.84.
- Update rdma-core version in manylinux wheels to 49.0.
- Make various updates to the Github Actions infrastructure.

.. rubric:: 4.1.1

- Add AVX and AVX-512 implementations for non-temporal memory copy. While this
  is transparent to the API, the Meson option names have changed to allow
  specific instruction sets to be disabled if necessary.

.. rubric:: 4.1.0

- Introduce :ref:`explicit start <py-explicit-start>` for receive streams.

.. rubric:: 4.0.2

Note that an oversight lead to some of the changes between 4.0.0b1 and 4.0.0
being omitted from 4.0.1. They are restored in 4.0.2.

- Fix type annotations for :py:class:`spead2.send.UdpStream` and
  :py:class:`spead2.send.asyncio.UdpStream`.
- Add more documentation for developers.
- Remove an old :file:`Makefile.am` that should have been removed in 4.0.0.
- Remove mocking of spead2 in readthedocs build.
- Change `.. code::` to `.. code-block::` in documentation.
- Simplify the implementation of :cpp:class:`!thread_pool_wrapper` and
  :cpp:class:`!buffer_reader` in Python binding code.
- Directly use pointer-to-member-functions in Python binding code.
- Test against numpy 1.26.0 (instead of 1.26.0rc1) on Python 3.12.

.. rubric:: 4.0.1

- Restore dependency on numpy, which was accidentally removed in 4.0.0.
- Change the ``test_numba`` extra to ``test-numba`` to normalise it in
  accordance with :pep:`685`.

.. rubric:: 4.0.0

This release makes major changes to the build system, and removes deprecated
features. See :doc:`migrate-4` for more detailed information about upgrading
and the deprecation removals.

Most of the changes are listed under 4.0.0b1 below. Since then, the following
changes have been made:

- Improve detection of gdrcopy by using the CUDA compiler to compile the
  test code.
- Remove ninja from ``build-system.requires``. If you have ninja installed on
  the system, that will be used for the Python install rather than
  downloading it.
- Make miscellaneous improvements to the build system.
- Remove an unused file (:file:`.ci/ccache-path.sh`).
- Work around a pytest bug to prevent tests running out of file descriptors
  (particularly on MacOS, which has a lower default limit).
- Add wheels for MacOS (Intel and Apple Silicon).
- Document that Meson must be at least 1.2.
- Make source paths in :file:`.debug` files more usable (relative to
  4.0.0b1).
- Remove :file:`.pyi` file entries for the functionality removed in 4.0.

.. rubric:: 4.0.0b1

This release makes major changes to the build system, and removes deprecated
features. See :doc:`migrate-4` for more detailed information about upgrading
and the deprecation removals.

- Replace the build system with Meson_.
- Update the C++ code to use C++17.
- No longer link against boost_system, and require Boost 1.69+.
- Remove generated code from release tarballs; some Python packages are now
  required at build time to build the C++ bindings.
- Fix an uninitialised variable that could cause a segmentation fault when
  using TCP to send and the initial connection failed.
- Fix a large number of compiler warnings that showed up after switching build
  systems (mainly related to unused function parameters and signed/unsigned
  comparisons).
- Fix the debug logging option so logging from inline functions in headers
  will also do debug logging without the user needing to make preprocessor
  defines.
- Fix :program:`spead2_bench.py` so that it functions when ibverbs support is
  not compiled in.
- Remove the need for boost_program_options to be installed to be able to
  install the Python bindings from source.
- Produce binary wheels for aarch64.
- Produce wheels for Python 3.12.
- Make numba and scipy optional for running tests (some tests will be skipped
  if they are not present).
- Update the libpcap embedded in the wheels to 1.10.4.
- Update the Boost version used to build wheels to 1.83.
- Update the rdma-core version used to build wheels to 47.0.
- Update the pybind11 build dependency to 2.11.1.
- Replace flake8 with ruff_ for linting.
- Remove the :file:`spead2/common_bind.h` header, which was unused.
- Remove the :c:macro:`!SPEAD2_DEPRECATED` macro.
- Remove build-time dependencies from :file:`requirements.txt`.
- Update the :file:`.pyi` files to use more modern syntax e.g., :pep:`585`,
  :pep:`604`, :pep:`613`.
- Replace references to nv_peer_mem with nvidia-peermem.
- Increase TTL of :program:`gpudirect_example` to 4.

.. _Meson: https://mesonbuild.com
.. _ruff: https://beta.ruff.rs/docs/

.. rubric:: 3.13.0

- Reformat the Python codebase using black_ and isort_.
- Add `pre-commit`_ configuration.
- On i386, check for SSE2 support at runtime rather than configure time.
- Free readers only when the stream is destroyed. This fixes a bug that caused
  the Python API to be accessed without the GIL when using
  :py:meth:`~spead2.recv.Stream.add_buffer_reader`.
- Improve unit tests by explicitly closing TCP sockets, to avoid
  :exc:`ResourceWarning` when testing with ``python -X dev``.
- Remove :py:mod:`wheel` from ``build-system.requires``.

.. _black: https://black.readthedocs.io/en/stable/
.. _isort: https://pycqa.github.io/isort/
.. _pre-commit: https://pre-commit.com/

.. rubric:: 3.12.0

- Add support for :doc:`recv-chunk-group` to assemble chunks in parallel.
- Simplify the way receive streams shut down. Users should not notice any
  change, but custom reader implementations will need to be updated.
- Update :meth:`!test_async_flush` and :meth:`!test_async_flush_fail` to keep
  handles to async tasks, to prevent them being garbage collected too early.
- Fix a bug where copying a :cpp:class:`spead2::recv::stream_config` would not
  deep copy the names of custom statistics, and so any statistics added to the
  copy would also affect the original, and there were also potential race
  conditions if a stream config was modified while holding stream statistics.
- Fix a bug (caused by the bug above) where passing a
  :cpp:class:`spead2::recv::stream_config` to construct a
  :cpp:class:`spead2::recv::chunk_stream` would modify the config. Passing
  the same config to construct two chunk streams would fail with an error.
- Fix the type annotation for the :py:class:`~.ChunkRingStream` constructor:
  the parameter name for `chunk_stream_config` was incorrect.
- Fix universal binary builds on MacOS (this was preventing Python 3.11 builds
  from succeeding).
- Fix :program:`spead2_bench.py`, which has failed to run at all for some time
  (possibly since 3.0).
- Avoid including Boost dynamic symbols in the Python module (helps reduce
  binary size).
- Strip static symbols out of the Python wheels (reduces size).
- Build Python wheels with link-time optimisation (small performance
  improvement).
- Python 3.8 is now the minimum version.

.. rubric:: 3.11.1

- Fix a packaging issue that meant automake and similar tools were required to
  compile (since 3.10).

.. rubric:: 3.11.0

- The chunking receiver is no longer experimental.
- The place callback for the chunking receiver can now provide extra data to be
  written to the chunk.

.. rubric:: 3.10.0

- Support pcap dumps that use the SLL format.
- Support a user-defined filter in the pcap file reader.
- Add experimental support for building a shared library.
- Assorted documentation updates

  - The SPEAD specification is now stored in the repository (the upstream
    link is broken).
  - Build PDFs on readthedocs.
  - Update the tuning documentation.

.. rubric:: 3.9.1

- Fix an :exc:`asyncio.InvalidStateError` that occurs when the future returned by
  :py:meth:`~.async_send_heap` or :py:meth:`~.async_send_heaps` is cancelled
  before it completes.

.. rubric:: 3.9.0

- Added ``substreams`` to :py:class:`spead2.recv.StreamConfig` to improve
  handling of interleaved heaps from multiple senders.
- Add libdivide to the dependencies.

.. rubric:: 3.8.0

- Drop support for Python 3.6, which has reached end-of-life.
- Test against Python 3.10 in Github Actions.
- Improve the accuracy of the rate limiter. Previously it could send
  slightly too fast due to rounding sleep times to whole numbers of
  nanoseconds.
- Eliminate dependence on distutils, which is deprecated in Python 3.10
  (#175).

.. rubric:: 3.7.0

- Add :py:const:`spead2.send.GroupMode.SERIAL`.
- Add :py:class:`spead2.send.HeapReferenceList`.
- Speed up C++ unit tests.
- Fix some spurious output in the statistics report from
  :program:`spead2_recv.py` (introduced in 3.5.0).
- Fix the help message from :program:`spead2_net_raw` to have the right name
  for the program.
- Update to latest version of pybind11.

.. rubric:: 3.6.0

- Allow a ringbuffer to be stopped only once the last producer has indicated
  completion, rather than the first.
- Change :py:class:`.ChunkRingStream` so that stops received from the network
  only shut down a shared ringbuffer once all the streams have stopped. A user
  call to ``stop`` will still stop the ringbuffer immediately.
- :py:meth:`.ChunkRingbuffer.stop` now returns a boolean to indicate whether
  this is the first time the ringbuffer was stopped.

.. rubric:: 3.5.0

- Add support for :ref:`custom-stats`.
- Change the allocate and ready callbacks on
  :cpp:class:`spead2::recv::chunk_stream` to take a pointer to the batch
  statistics. This is a **backwards-incompatible change** (keep in mind that
  chunking receive is still experimental). Code that uses
  :cpp:class:`spead2::recv::chunk_ring_stream` is unaffected.
- Change the design of deleters for
  :cpp:class:`spead2::memory_allocator`. Code that calls ``get_user`` or
  ``get_deleter`` on a pointer allocated by spead2 may now get a ``nullptr``
  back. Code that uses a custom memory allocator and that calls these
  functions on pointers allocated by that allocator should continue to work.
- Allow a ready callback to be used together with
  :cpp:class:`spead2::recv::chunk_ring_stream`, to finish preparation of a
  chunk before it pushed to the ringbuffer.
- In Python, avoid copying immediate items when given as 0-d arrays with dtype
  ``>u8``. This makes it practical to pre-define heaps and later update their
  values rather than creating new heap objects.
- Make :py:class:`spead2.send.Stream`, :py:class:`spead2.send.SyncStream` and
  :py:class:`spead2.send.asyncio.AsyncStream` available for type annotations.
- Fix an occasional segfault when stopping a
  :py:class:`spead2.recv.ChunkRingStream`.

.. rubric:: 3.4.0

- Add :doc:`recv-chunk`.
- Add missing :py:meth:`spead2.recv.Stream.add_udp_pcap_file_reader` to .pyi file.
- Add :py:meth:`spead2.InprocQueue.add_packet`.
- Prevent conversions from ``None`` to :py:class:`spead2.ThreadPool`.

.. rubric:: 3.3.2

- :cpp:class:`spead2::recv::mem_reader` now stops the stream gracefully,
  allowing incomplete heaps to be flushed.

.. rubric:: 3.3.1

- Convert :program:`spead2_net_raw` to a C++ file so that it gets the same
  compiler flags as everything else.
- Migrate from Travis CI to Github Actions.
- Fix some warnings generated by Clang.
- Fix some test failures with PyPy.

.. rubric:: 3.3.0

- Add :ref:`spead2_net_raw` tool.
- Eliminate some compiler warnings about unused parameters.
- Update build process to use pypa-build and setuptools_scm.
- Update to pybind11 2.6.2.

.. rubric:: 3.2.2

- Use ``python3`` instead of ``python`` to invoke Python (so that it works
  even on systems where ``python`` is absent or is Python 2).
- Work around a bug that prevented compilation on Boost 1.76.

.. rubric:: 3.2.1

- Update type annotations to use :class:`numpy.typing.DTypeLike` for dtype
  arguments, to prevent false warnings from mypy.

.. rubric:: 3.2.0

- Add :cpp:func:`spead2::recv::heap::get_payload` to allow the payload
  pointer to be retrieved from a complete heap.
- Make the ibverbs sender compatible with `PeerDirect`_.
- Add examples programs showing integration with `gdrcopy`_ and
  `PeerDirect`_.
- Always use SFENCE at end of :cpp:func:`memcpy_nontemporal` so that it is
  appropriate for use with `gdrcopy`_.
- Fix a memory leak when receiving with ibverbs.

.. _gdrcopy: https://github.com/NVIDIA/gdrcopy
.. _PeerDirect: https://docs.mellanox.com/pages/viewpage.action?pageId=32413288

.. rubric:: 3.1.3

- Fix installation of header files: some newer headers were not being
  installed, breaking builds for C++ projects.

.. rubric:: 3.1.2

- Fix a use-after-free bug that could cause a crash when freeing a send
  stream.
- Improve send performance by eliminating a memory allocation from packet
  generation.

.. rubric:: 3.1.1

- Set ``IBV_ACCESS_RELAXED_ORDERING`` flag on ibverbs memory regions. This
  reduces packet loss in some circumstances (observed on Epyc 2 system with
  lots of memory traffic).

.. rubric:: 3.1.0

- Add :py:meth:`~spead2.send.AbstractStream.send_heaps` and
  :py:meth:`~spead2.send.asyncio.AbstractStream.async_send_heaps` to send
  groups of heaps with interleaved packets.
- Upgrade to pybind11 2.6.0, which contains a workaround for a bug in CPython
  3.9.0.

.. rubric:: 3.0.1

- Bring the type stubs up to date.
- Fix a typo in the documentation.

.. rubric:: 3.0.0

Version 3.0 contains a number of breaking API changes. For information on
updating your existing code, refer to :doc:`migrate-3`.

The :doc:`ibverbs <py-ibverbs>` acceleration has been substantially modified to use a
newer version of rdma-core. It will no longer compile against versions of
MLNX-OFED prior to 5.0. Compiled code (such as Python wheels) will still run
against old versions of MLNX-OFED, but extension features such as multi-packet
receive queues and packet timestamps will not work, and nor will
:program:`mcdump`. It is recommended that if you are using ibverbs acceleration
with older MLNX-OFED drivers that you stick with spead2 2.x until you're able
to upgrade the drivers and spead2 simultaneously.

- Support multiple "substreams" in a send stream (see :ref:`py-substreams`).
- Reduce overhead for dealing with incomplete heaps.
- Allow ibverbs senders to register memory regions for zero-copy
  transmission.
- Add C++ preprocessor defines for the version number.
- Use IP/UDP checksum offloading for sending with ibverbs (improves
  performance and also adds UDP checksum which is otherwise omitted).
- Add wheels for Python 3.9.
- Drop support for Python 3.5, which is end-of-life.
- Change code examples to use standard SPEAD rather than PySPEAD bug
  compatibility.
- Change :cpp:class:`spead2::send::streambuf_stream` so that when the
  streambuf only partially writes a packet, the partial byte count is
  included in the count returned to the callback.
- :cpp:func:`spead2::send::stream::flush` now only blocks until the
  previously enqueued heaps are completed. Another thread that keeps adding
  heaps would previously have prevented it from returning.
- Partially rewrite the sending infrastructure, resulting in performance
  improvements, in some cases of over 10%.
- Setting a buffer size of 0 for a :py:class:`~spead2.send.UdpIbvStream` now
  uses the default buffer size, instead of a 1-packet buffer.
- Fix :program:`spead2_bench.py` ignoring the :option:`!--send-affinity` option.
- Add :option:`!--verify` option to :program:`spead2_send` and
  :program:`spead2_recv` to aid in testing the code. To support this,
  :program:`spead2_send` was modified so that each in-flight heap uses
  different memory, which may reduce performance (due to less cache re-use)
  even when the option is not given.
- Miscellaneous performance improvements.
- Support hardware send rate limiting when using ibverbs (disabled by default).
- Discover libibverbs and pcap using pkg-config where possible.
- Make :program:`configure` print out the configuration that will be compiled.
- Update the Python wheels to use manylinux2014. This uses a newer compiler
  (potentially giving better performance) and supports :c:func:`sendmmsg`.
- A number of deprecated functions have been removed.
- Avoid ibverbs code creating a send queue for receiver or vice versa.
- Rename ``slave`` option to :program:`spead2_bench` to ``agent``.

Compared to 3.0.0b2 there is a critical bug fix for a race condition in the
send code.

.. rubric:: 3.0.0b2

Version 3.0 contains a number of breaking API changes. For information on
updating your existing code, refer to :doc:`migrate-3`.

Other changes:

- Support multiple "substreams" in a send stream (see :ref:`py-substreams`).
- Reduce overhead for dealing with incomplete heaps.
- Allow ibverbs senders to register memory regions for zero-copy
  transmission.
- Add C++ preprocessor defines for the version number.
- Use IP/UDP checksum offloading for sending with ibverbs (improves
  performance and also adds UDP checksum which is otherwise omitted).
- Drop support for Python 3.5, which is end-of-life.
- Change code examples to use standard SPEAD rather than PySPEAD bug
  compatibility.
- Change :cpp:class:`spead2::send::streambuf_stream` so that when the
  streambuf only partially writes a packet, the partial byte count is
  included in the count returned to the callback.
- :cpp:func:`spead2::send::stream::flush` now only blocks until the
  previously enqueued heaps are completed. Another thread that keeps adding
  heaps would previously have prevented it from returning.
- Partially rewrite the sending infrastructure, resulting in performance
  improvements, in some cases of over 10%.
- Setting a buffer size of 0 for a :py:class:`~spead2.send.UdpIbvStream` now
  uses the default buffer size, instead of a 1-packet buffer.
- Fix :program:`spead2_bench.py` ignoring the :option:`!--send-affinity` option.
- The hardware rate limiting introduced in 3.0.0b1 is now disabled by default,
  as it proved to be significantly less accurate than the software rate limiter
  in some cases. The interface has also been changed from a boolean to an enum
  (with the default being ``AUTO``) so that it can later be re-enabled under
  circumstances where it is known to work well, while still allowing it to be
  explicitly enabled or disabled.
- Add :option:`!--verify` option to :program:`spead2_send` and
  :program:`spead2_recv` to aid in testing the code. To support this,
  :program:`spead2_send` was modified so that each in-flight heap uses
  different memory, which may reduce performance (due to less cache re-use)
  even when the option is not given.
- Miscellaneous performance improvements.

Additionally, refer to the changes for 3.0.0b1 below.

.. rubric:: 3.0.0b1

The :doc:`ibverbs <py-ibverbs>` acceleration has been substantially modified to use a
newer version of rdma-core. It will no longer compile against versions of
MLNX-OFED prior to 5.0. Compiled code (such as Python wheels) will still run
against old versions of MLNX-OFED, but extension features such as multi-packet
receive queues and packet timestamps will not work. It is recommended that if
you are using ibverbs acceleration with older MLNX-OFED drivers that you stick
with spead2 2.x until you're able to upgrade the drivers and spead2
simultaneously.

Other changes:

- Support hardware send rate limiting when using ibverbs.
- Discover libibverbs and pcap using pkg-config where possible.
- Make :program:`configure` print out the configuration that will be compiled.
- Update the Python wheels to use manylinux2014. This uses a newer compiler
  (potentially giving better performance) and supports :c:func:`sendmmsg`.
- Add wheels for Python 3.9.
- A number of deprecated functions have been removed.
- Avoid ibverbs code creating a send queue for receiver or vice versa.
- Rename ``slave`` option to :program:`spead2_bench` to ``agent``.

.. rubric:: 2.1.2

- Make verbs acceleration work when run against MLNX OFED 5.x, including with
  Python wheels. Note that it will not use multi-packet receive queues, so
  receive performance may still be better on MLNX OFED 4.9.

.. rubric:: 2.1.1

- Update pybind to 2.5.0.
- Fix compilation against latest rdma-core.
- Some documentation cleanup.

.. rubric:: 2.1.0

- Support unicast receive with ibverbs acceleration (including in
  :program:`mcdump`).
- Fix :program:`spead2_recv` listening only on loopback when given just a port
  number.
- Support unicast addresses in a few APIs that previously only accepted
  multicast addresses; in most cases the unicast address must match the
  interface address.
- Add missing ``<map>`` include to ``<spead2/recv_heap.h>``.
- Show the values of immediate items in :program:`spead2_recv`.
- Fix occasional crash when using thread pool with more than one thread
  together with ibverbs.
- Fix bug in mcdump causing it to hang if the arguments couldn't be parsed
  (only happened when capturing to file).
- Fix :program:`spead2_recv` reporting statistics that may miss out the last
  batch of packets.

.. rubric:: 2.0.2

- Log warnings on some internal errors (that hopefully never happen).
- Include wheels for Python 3.8.
- Build debug symbols for binary wheels (in a separate tarball on Github).

.. rubric:: 2.0.1

- Fix race condition in TCP receiver (#78).
- Update vendored pybind11 to 2.4.2.

.. rubric:: 2.0.0

- Drop support for Python 2.
- Drop support for Python 3.4.
- Drop support for trollius.
- Drop support for netmap.
- Avoid creating some cyclic references. These were not memory leaks, but
  prevented CPython from freeing objects as soon as it might have.
- Update vendored pybind11 to 2.4.1.

.. rubric:: 1.14.0

- Add `new_order` argument to :py:meth:`spead2.ItemGroup.update`.
- Improved unit tests.

.. rubric:: 1.13.1

- Raise :exc:`ValueError` on a dtype that has zero itemsize (#37).
- Change exception when dtype has embedded objects from :exc:`TypeError` to
  :exc:`ValueError` for consistency
- Remove duplicated socket handle in UDP receiver (#67).
- Make `max_poll` argument to :py:class:`spead2.send.UdpIbvStream` actually
  have an effect (#55).
- Correctly report EOF errors in :cpp:class:`spead2::send::streambuf_stream`.
- Wrap implicitly computed heap cnts to the number of available bits (#3).
  Previously behaviour was undefined.
- Some header files were not installed by ``make install`` (#72).

.. rubric:: 1.13.0

- Significant performance improvements to send code (in some cases an order of
  magnitude improvement).
- Add :option:`!--max-heap` option to :program:`spead2_send` and
  :program:`spead2_send.py` to control the depth of the send queue.
- Change the meaning of the :option:`!--heaps` option in :program:`spead2_bench`
  and :program:`spead2_bench.py`: it now also controls the depth of the sending
  queue.
- Fix a bug in send rate limiting that could allow the target rate to be
  exceeded under some conditions.
- Remove :option:`!--threads` option from C++ :program:`spead2_send`, as the new
  optimised implementation isn't thread-safe.
- Disable the ``test_numpy_large`` test on macOS, which was causing frequent
  failures on TravisCI due to dropped packets.

.. rubric:: 1.12.0

- Provide manylinux2010 wheels.
- Dynamically link to libibverbs and librdmacm on demand. This allows binaries
  (particularly wheels) to support verbs acceleration but still work on systems
  without these libraries installed.
- Support for Boost 1.70. Unfortunately Boost 1.70 removes the ability to query
  the io_service from a socket, so constructors that take a socket but no
  io_service are omitted when compiling with Boost 1.70 or newer.
- Fix some compiler warnings from GCC 8.

.. rubric:: 1.11.4

- Rework the locking internals of :cpp:class:`spead2::recv::stream` so that
  a full ringbuffer doesn't block new readers from being added. This changes
  the interfaces between :cpp:class:`spead2::recv::reader` and
  :cpp:class:`spead2::recv::stream_base`, but since users generally don't deal
  with that interface the major version hasn't been incremented.
- Fix a spurious log message if an in-process receiver is manually stopped.
- Fix an intermittent unit test failure due to timing.

.. rubric:: 1.11.3

- Undo the optimisation of using a single flow steering rule to cover multiple
  multicast groups (see #11).

.. rubric:: 1.11.2

- Fix ``-c`` option to :program:`mcdump`.
- Fix a missing ``#include`` that could be exposed by including headers in a
  particular order.
- Make :cpp:class:`spead2::recv::heap`'s move constructor and move assignment
  operator ``noexcept``.
- Add a `long_description` to the Python metadata.

.. rubric:: 1.11.1

- Update type stubs for new features in 1.11.0.

.. rubric:: 1.11.0

- Add :py:attr:`spead2.recv.Stream.allow_unsized_heaps` to support rejecting
  packets without a heap length.
- Add extended custom memcpy support (C++ only) for scattering data from
  packets.

.. rubric:: 1.10.1

- Use ibverbs multi-packet receive queues automatically when available
  (supported by mlx5 driver).
- Automatically reduce buffer size for verbs receiver to match hardware limits
  (fixed #64).
- Gracefully handle Ctrl-C in :program:`spead2_recv` and print statistics.
- Add typing stub files to assist checking with Mypy.
- Give a name to the argument of
  :py:meth:`spead2.recv.Stream.add_inproc_reader`.
- Fix Python binding for one of the UDP reader overloads that takes an existing
  socket. This was a deprecated overload.
- Add a unit test for ibverbs support. It's not run by default because it
  needs specific hardware.

.. rubric:: 1.10.0

- Accelerate per-packet processing, particularly when `max_heaps` is large.
- Accelerate per-heap processing, particularly for heaps with few items.
- Add a fast path for single-packet heaps.
- Improve performance of the pcap reader by working on batches of packets.
- Provide access to ringbuffer size and capacity for diagnostics.
- Add extra fields to :py:class:`spead2.recv.StreamStats`.
- Add support for pcap files to the C++ version of :program:`spead2_recv`.
- Update the vendored pybind11 to 2.2.4 (fixes some warnings on Python 3.7).
- Deprecate netmap support in documentation.

.. rubric:: 1.9.2

- autotools are no longer required to install the C++ build (when installing
  from a release tarball).

.. rubric:: 1.9.1

- Make :py:meth:`spead2.recv.asyncio.Stream.get` always yield to the event loop
  even if there is a heap ready.
- Avoid :py:meth:`spead2.recv.asyncio.Stream.get` holding onto a reference to
  the heap (via a future) for longer than necessary.

.. rubric:: 1.9.0

- Add support for TCP/IP (contributed by Rodrigo Tobar).
- Changed command-line options for
  :program:`spead2_send`/:program:`spead2_recv`: :option:`!--ibv` and
  :option:`!--netmap` are now boolean flags, and the interface address is set
  with :option:`!--bind`.
- Added option to specify interface address for
  :cpp:class:`spead2::send::udp_stream` even when not using the multicast
  constructors.
- Constructors that take an existing socket now expect the user to set all
  socket options. The old versions that take a socket buffer size are
  deprecated. Note that the behaviour of :cpp:class:`spead2::send::udp_stream`
  with a socket has **changed**: if no buffer size is given, it is left at the
  OS default, rather than applying the spead2 default.
- Fix a bug causing undefined behaviour if a send class is destroyed while
  there is still data in flight.

.. rubric:: Version 1.8.0

- Add :doc:`py-inproc`
- Fix unit testing on Python 3.7
- Add :cpp:func:`spead2::send::heap::get_item`
- Support asynchronous iterator protocol for
  :py:class:`spead2.recv.asyncio.Stream` (in Python 3.5+).

.. rubric:: Version 1.7.2

- Add progress reports to mcdump
- Add ability to pass ``-`` as filename to mcdump to skip file writing.
- Add :option:`!--count` option to mcdump

.. rubric:: Version 1.7.1

There are no code changes, but this release fixes a packaging error in 1.7.0
that prevented the asyncio integration from being included.

.. rubric:: Version 1.7.0

- Support for pcap files. Files passed to :program:`spead2_recv.py` are now
  assumed to be pcap files, rather than raw concatenated packets.
- Only log warnings about the ringbuffer being full if at least one stream
  reader is lossy (indicated by a new virtual member function in
  :cpp:class:`spead2::recv::Reader`).

.. rubric:: Version 1.6.0

- Change :program:`spead2_send.py` and :program:`spead2_send` to interpret
  the :option:`!--rate` option as Gb/s and not Gib/s.
- Change send rate limiting to bound the rate at which we catch up if we fall
  behind. This is controlled by a new attribute of
  :class:`~spead2.send.StreamConfig`.
- Add report at end of :program:`spead2_send.py` and :program:`spead2_send`
  on the actual number of bytes sent and achieved rate.
- Fix a race condition where the stream statistics might only be updated after
  the stream ended (which lead to unit test failures in some cases).

.. rubric:: Version 1.5.2

- Report statistics when :program:`spead2_recv.py` is stopped by SIGINT.
- Add --ttl option to :program:`spead2_send.py` and :program:`spead2_send`.

.. rubric:: Version 1.5.1

- Explicitly set UDP checksum to 0 in IBV sender, instead of leaving
  arbitrary values.
- Improved documentation of asyncio support.

.. rubric:: Version 1.5.0

- Support for asyncio in Python 3. For each trollius module there is now an
  equivalent asyncio module. The installed utilities use asyncio on Python
  3.4+.
- Add :attr:`spead2.recv.Stream.stop_on_stop_item` to allow a stream to keep
  receiving after a stop item is received.
- Switch shutdown code to use atexit instead of a capsule destructor, to
  support PyPy.
- Test PyPy support with Travis.

.. rubric:: Version 1.4.0

- Remove :option:`!--bind` option to :program:`spead2_recv.py` and :program:`spead2_recv`.
  Instead, use :samp:`{host}:{port}` as the source. This allows subscribing to
  multiple multicast groups.
- Improved access to information about incomplete heaps
  (:py:class:`spead2.recv.IncompleteHeap` type).
- Add :py:attr:`.MemoryPool.warn_on_empty` control.
- Add warning when a stream ringbuffer is full.
- Add statistics to streams.
- Fix spead2_send.py to send a stop heap when using :option:`!--heaps`. It was
  acccidentally broken in 1.2.0.
- Add support for packet timestamping in mcdump.
- Return the previous logging function from :cpp:func:`spead2::set_log_function`.
- Make Python logging from C++ code asynchronous, to avoid blocking the thread pool
  on the GIL.
- Upgrade to pybind11 2.2.1 internally.
- Some fixes for PyPy support.

.. rubric:: Version 1.3.2

- Fix segfault in shutdown for :file:`spead2_recv.py` (fixes #56).
- Fix for :py:exc:`TypeError` in Python 3.6 when reading fields that aren't
  aligned to byte boundaries.
- Include binary wheels in releases.

.. rubric:: Version 1.3.1

- Fix multi-endpoint form of
  :py:meth:`spead2.recv.Stream.add_udp_ibv_reader`.

.. rubric:: Version 1.3.0

- Rewrite the Python wrapping using pybind11. This should not cause any
  compatibility problems, unless you're using the :file:`spead2/py_*.h`
  headers.
- Allow passing :cpp:class:`std::shared_ptr<thread_pool>` to constructors that
  take a thread pool, with the constructed object holding a reference.
- Prevent constructing a :py:class:`spead2.recv.Stream` with
  ``max_heaps=0`` (fixes #54).

.. rubric:: Version 1.2.2

- Fix rate limiting causing longer sleeps than necessary (fixes #53).

.. rubric:: Version 1.2.1

- Disable LTO by default and require the user to opt in, because even if the
  compiler supports it, linking can still fail (fixes #51).

.. rubric:: Version 1.2.0

- Support multiple endpoints for one :cpp:class:`~spead2::recv::udp_ibv_reader`
  (fixes #48).

- Fix compilation on OS X 10.9 (fixes #49)

- Fix :cpp:func:`spead2::ringbuffer<T>::emplace` and :cpp:func:`spead2::ringbuffer<T>::try_emplace`

- Improved error messages when passing invalid arguments to mcdump

.. rubric:: Version 1.1.2

- Only log descriptor replacement if it actually replaces an existing name or
  ID (regression in 1.1.1).
- Fix build on ARM where compiling against asio requires linking against
  pthread.
- Updated and expanded performance tuning guide.

.. rubric:: Version 1.1.1

- Report the item name in exception for "too few elements for shape" errors
- Overhaul of rules for handling item descriptors that change the name or ID
  of an item. This prevents stale items from hanging around when the sender
  changes the name of an item but keeps the same ID, which can cause unrelated
  errors on the receiver if the shape also changes.

.. rubric:: Version 1.1.0

- Allow heap cnt to be set explicitly by sender, and the automatic heap cnt
  sequence to be specified as a start value and step.

.. rubric:: Version 1.0.1

- Fix exceptions to include more information about the source of the failure
- Add :ref:`mcdump` tool

.. rubric:: Version 1.0.0

- The C++ API installation has been changed to use autoconf and automake. As a
  result, it is possible to run ``make install`` and get the static library,
  headers, and tools installed.
- The directory structure has changed. The :file:`spead2_*` tools are now
  installed, example code is now in the :file:`examples` directory, and the
  headers have moved to :file:`include/spead2`.
- Add support for sending data using libibverbs API (previously only supported
  for receiving)
- Fix async_send_heap (in Python) to return a future instead of being a
  coroutine: this fixes a problem with undefined ordering in the trollius
  example.
- Made sending streams polymorphic, with abstract base class
  :cpp:class:`spead2::send::stream`, to simplify writing generic code that can
  operate on any type of stream. This will **break** code that depended on the
  old template class of the same name, which has been renamed to
  :cpp:class:`spead2::send::stream_impl`.
- Add :option:`!--memcpy-nt` to :program:`spead2_recv.py` and
  :program:`spead2_bench.py`
- Multicast support in :program:`spead2_bench.py` and :program:`spead2_bench`
- Changes to the algorithm for :program:`spead2_bench.py` and
  :program:`spead2_bench`: it now starts by computing the maximum send speed,
  and then either reporting that this is the limiting factor, or using it to
  start the binary search for the receive speed. It is also stricter about
  lost heaps.
- Some internal refactoring of code for dealing with raw packets, so that it
  is shared between the netmap and ibv readers.
- Report function name that failed in semaphore system_error exceptions.
- Make the unit tests pass on OS X (now tested on travis-ci.org)

.. rubric:: Version 0.10.4

- Refactor some of the Boost.Python glue code to make it possible to reuse
  parts of it in writing new Python extensions that use the C++ spead2 API.

.. rubric:: Version 0.10.3

- Suppress "operation aborted" warnings from UDP reader when using the API
  to stop a stream (introduced in 0.10.0).
- Improved elimination of duplicate item pointers, removing them as they're
  received rather than when freezing a live heap (fixes #46).
- Use hex for reporting item IDs in log messages
- Fix reading from closed file descriptor after stream.stop() (fixes #42)
- Fix segmentation fault when using ibverbs but trying to bind to a
  non-RDMA device network interface (fixes #45)

.. rubric:: Version 0.10.2

- Fix a performance problem when a heap contains many packets and every
  packet contains item pointers. The performance was quadratic instead of
  linear.

.. rubric:: Version 0.10.1

- Fixed a bug in registering `add_udp_ibv_reader` in Python, which broke
  :program:`spead2_recv.py`, and possibly any other code using this API.
- Fixed :program:`spead2_recv.py` ignoring :option:`!--ibv-max-poll` option

.. rubric:: Version 0.10.0

- Added support for libibverbs for improved performance in both :doc:`Python
  <py-ibverbs>` and :doc:`C++ <cpp-ibverbs>`.

- Avoid per-packet shared_ptr reference counting, accidentally introduced in
  0.9.0, which caused a small performance regression. This is unfortunately a
  **breaking** change to the interface for implementing custom memory
  allocators.

.. rubric:: Version 0.9.1

- Fix using a :py:class:`~spead2.MemoryPool` with a thread pool and low water
  mark (regression in 0.9.0).

.. rubric:: Version 0.9.0

- Add support for custom memory allocators.

.. rubric:: Version 0.8.2

- Ensure correct operation when `loop=None` is passed explicitly to trollius
  stream constructors, for consistency with functions that have it as a keyword
  parameter.

.. rubric:: Version 0.8.1

- Suppress ``recvmmsg: resource temporarily unavailable`` warnings (fixes #43)

.. rubric:: Version 0.8.0

- Extend :py:class:`~spead2.MemoryPool` to allow a background thread to
  replenish the pool when it gets low.
- Extend :py:class:`~spead2.ThreadPool` to allow the user to pin the threads to
  specific CPU cores (on glibc).

.. rubric:: Version 0.7.1

- Fix ring_stream destructor to not deadlock (fixes #41)

.. rubric:: Version 0.7.0

- Change handling of incomplete heaps (fixes #39). Previously, incomplete heaps
  were only abandoned once there were more than `max_heaps` of them. Now, they
  are abandoned once `max_heaps` more heaps are seen, even if those heaps were
  complete. This causes the warnings for incomplete heaps to appear closer to
  the time they arrived, and also has some extremely small performance
  advantages due to changes in the implementation.

- **backwards-incompatible change**: remove
  :py:meth:`~spead2.recv.Stream.set_max_heaps`. It was not previously
  documented, so hopefully is not being used. It could not be efficiently
  supported with the design changes above.

- Add :py:meth:`spead2.recv.Stream.set_memcpy` to control non-temporal caching
  hints.

- Fix C++ version of spead2_bench to actually use the memory pool

- Reduce memory usage in spead2_bench (C++ version)

.. rubric:: Version 0.6.3

- Partially fix #40: :py:meth:`~spead2.recv.Stream.set_max_heaps` and
  :py:meth:`~spead2.recv.Stream.set_memory_pool` will no longer deadlock if
  called on a stream that has already had a reader added and is receiving
  data.

.. rubric:: Version 0.6.2

- Add a fast path for integer items that exactly fit in an immediate.

- Optimise Python code by replacing np.product with a pure Python
  implementation.

.. rubric:: Version 0.6.1

- Filter out duplicate items from a heap. It is undefined which of a set of
  duplicates will be retained (it was already undefined for
  :py:class:`spead2.ItemGroup`).

.. rubric:: Version 0.6.0

- Changed item versioning on receive to increment version number on each update
  rather that setting to heap id. This is more robust to using a single item
  or item group with multiple streams, and most closely matches the send path.
- Made the protocol enums from the C++ library available in the Python library
  as well.
- Added functions to create stream start items (send) and detect them (recv).

.. rubric:: Version 0.5.0

- Added friendlier support for multicast. When a multicast address is passed
  to :py:meth:`~spead2.recv.Stream.add_udp_reader`, the socket will
  automatically join the multicast group and set :cpp:var:`SO_REUSEADDR` so
  that multiple sockets can consume from the same stream. There are also new
  constructors and methods to give explicit control over the TTL (send)
  and interface (send and receive), including support for IPv6.

.. rubric:: Version 0.4.7

- Added in-memory mode to the C++ version of spead2_bench, to measure the
  packet handling speed independently of the lossy networking code
- Optimization to duplicate packet checks. This makes a substantial
  performance improvement when using small (e.g. 512 byte) packets and large
  heaps.

.. rubric:: Version 0.4.6

- Fix a data corruption (use-after-free) bug on send side when data is being
  sent faster than the socket can handle it.

.. rubric:: Version 0.4.5

- Fix bug causing some log messages to be remapped to DEBUG level

.. rubric:: Version 0.4.4

- Increase log level for packet rejection from DEBUG to INFO

- Some minor optimisations

.. rubric:: Version 0.4.3

- Handle heaps that have out-of-range item offsets without crashing (#32)

- Fix handling of heaps without heap length headers

- :py:meth:`spead2.send.UdpStream.send_heap` now correctly raises
  :py:exc:`IOError` if the heap is rejected due to being full, or if there was
  an OS-level error in sending the heap.

- Fix :py:meth:`spead2.send.trollius.UdpStream.async_send_heap` for the case
  where the last sent heap failed.

- Use :manpage:`eventfd(2)` for semaphores on Linux, which makes a very small
  improvement in ringbuffer performance.

- Prevent messages about descriptor replacements for descriptor reissues with
  no change.

- Fix a use-after-free bug (affecting Python only).

- Throw :py:exc:`OverflowError` on out-of-range UDP port number, instead of
  wrapping.

.. rubric:: Version 0.4.2

- Fix compilation on systems without glibc

- Fix test suite for non-Linux systems

- Add :py:meth:`spead2.send.trollius.UdpStream.async_flush`

.. rubric:: Version 0.4.1

- Add C++ version of spead2_recv, a more fully-featured alternative to test_recv

- **backwards-incompatible change**:
  Add `ring_heaps` parameter to :cpp:class:`~spead2::recv::ring_stream`
  constructor. Code that specifies the
  `contiguous_only` parameter will need to be
  modified since the position has changed. Python code is unaffected.

- Increased the default for `ring_heaps` from 2 (previously hardcoded) to 4 to
  improve throughput for small heaps.

- Add support for user to provide the socket for UDP communications. This
  allows socket options to be set by the user, for example, to configure
  multicast.

- Force numpy>=1.9.2 to avoid a numpy [bug](https://github.com/numpy/numpy/issues/5356).

- Add experimental support for receiving packets via netmap

- Improved receive performance on Linux, particularly for small packets, using
  [recvmmsg](http://linux.die.net/man/2/recvmmsg).

.. rubric:: Version 0.4.0

- Enforce ASCII encoding on descriptor fields.

- Warn if a heap is dropped due to being incomplete.

- Add --ring option to C++ spead2_bench to test ringbuffer performance.

- Reading from a memory buffer (e.g. with
  :py:func:`~spead2.recv.Stream.add_buffer_reader`) is now reliable, instead of
  dropping heaps if the consumer doesn't keep up (heaps can still be dropped if
  packets extracted from the buffer are out-of-order, but it is
  deterministic).

- The receive ringbuffer now has a fixed size (2), and pushes are blocking. The
  result is lower memory usage, and it is no longer necessary to pass a large
  `max_heaps` value to deal with the consumer not always keeping up. Instead,
  it may be necessary to increase the socket buffer size.

- **backwards-incompatible change**:
  Calling :cpp:func:`spead2::recv::ring_stream::stop` now discards remaining
  partial heaps instead of adding them to the ringbuffer. This only affects the
  C++ API, because the Python API does not provide any access to partial heaps
  anyway.

- **backwards-incompatible change**:
  A heap with a stop flag is swallowed rather than passed to
  :cpp:func:`~spead2::recv::stream::heap_ready` (see issue
  [#29](https://github.com/ska-sa/spead2/issues/29)).

.. rubric:: Version 0.3.0

This release contains a number of backwards-incompatible changes in the Python
bindings, although most uses will probably not notice:

- When a received character array is returned as a string, it is now of type
  :py:class:`str` (previously it was :py:class:`unicode` in Python 2).

- An array of characters with a numpy descriptor with type `S1` will no longer
  automatically be turned back into a string. Only using a format of
  `[('c', 8)]`  will do so.

- The `c` format code may now only be used with a length of 8.

- When sending, values will now always be converted to a numpy array first,
  even if this isn't the final representation that will be put on the network.
  This may lead to some subtle changes in behaviour.

- The `BUG_COMPAT_NO_SCALAR_NUMPY` introduced in 0.2.2 has been removed. Now,
  specifying an old-style format will always use that format at the protocol
  level, rather than replacing it with a numpy descriptor.

There are also some other bug-fixes and improvements:

- Fix incorrect warnings about send buffer size.

- Added --descriptors option to spead2_recv.py.

- The `dtype` argument to :py:meth:`spead2.ItemGroup.add_item` is now
  optional, removing the need to specify `dtype=None` when passing a format.

.. rubric:: Version 0.2.2

- Workaround for a PySPEAD bug that would cause PySPEAD to fail if sent a
  simple scalar value. The user must still specify scalars with a format
  rather than a dtype to make things work.

.. rubric:: Version 0.2.1

- Fix compilation on OS X again. The extension binary will be slightly larger as
  a result, but still much smaller than before 0.2.0.

.. rubric:: Version 0.2.0

- **backwards-incompatible change**: for sending, the heap count is now tracked
  internally by the stream, rather than an attribute of the heap. This affects
  both C++ and Python bindings, although Python code that always uses
  :py:class:`~spead2.send.HeapGenerator` rather than directly creating heaps
  will not be affected.

- The :py:class:`~spead2.send.HeapGenerator` is extended to allow items to be
  added to an existing heap and to give finer control over whether descriptors
  and/or values are put in the heap.

- Fixes a bug that caused some values to be cast to non-native endian.

- Added overloaded equality tests on Flavour objects.

- Strip the extension binary to massively reduce its size

.. rubric:: Version 0.1.2

- Coerce values to int for legacy 'u' and 'i' fields

- Fix flavour selection in example code

.. rubric:: Version 0.1.1

- Fixes to support OS X

.. rubric:: Version 0.1.0

- First public release
