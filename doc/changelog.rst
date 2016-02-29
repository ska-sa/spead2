Changelog
=========

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
