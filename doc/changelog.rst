Changelog
=========

.. rubric:: 1.13.0

- Significant performance improvements to send code (in some cases an order of
  magnitude improvement).
- Add :option:`--max-heap` option to :program:`spead2_send` and
  :program:`spead2_send.py` to control the depth of the send queue.
- Change the meaning of the :option:`--heaps` option in :program:`spead2_bench`
  and :program:`spead2_bench.py`: it now also controls the depth of the sending
  queue.
- Fix a bug in send rate limiting that could allow the target rate to be
  exceeded under some conditions.
- Remove :option:`--threads` option from C++ :program:`spead2_send`, as the new
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
  :program:`spead2_send`/:program:`spead2_recv`: :option:`--ibv` and
  :option:`--netmap` are now boolean flags, and the interface address is set
  with :option:`--bind`.
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
- Add :option:`--count` option to mcdump

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
  the :option:`--rate` option as Gb/s and not Gib/s.
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

- Remove :option:`--bind` option to :program:`spead2_recv.py` and :program:`spead2_recv`.
  Instead, use :samp:`{host}:{port}` as the source. This allows subscribing to
  multiple multicast groups.
- Improved access to information about incomplete heaps
  (:py:class:`spead2.recv.IncompleteHeap` type).
- Add :py:attr:`.MemoryPool.warn_on_empty` control.
- Add warning when a stream ringbuffer is full.
- Add statistics to streams.
- Fix spead2_send.py to send a stop heap when using :option:`--heaps`. It was
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
- Add :option:`--memcpy-nt` to :program:`spead2_recv.py` and
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
- Fixed :program:`spead2_recv.py` ignoring :option:`--ibv-max-poll` option

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
