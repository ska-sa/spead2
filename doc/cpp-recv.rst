Receiving
=========

Heaps
-----
Unlike the Python bindings, the C++ bindings expose two heap types: *live heaps*
(:cpp:class:`spead2::recv::live_heap`) are used for heaps being constructed,
and may be missing data; *frozen heaps* (:cpp:class:`spead2::recv::heap`)
always have all their data. Frozen heaps can be move-constructed from live
heaps, which will typically be done in the callback.

.. doxygenclass:: spead2::recv::live_heap
   :members: is_complete,is_contiguous,is_end_of_stream,get_cnt,get_bug_compat

.. doxygenclass:: spead2::recv::heap
   :members:

.. doxygenstruct:: spead2::recv::item
   :members:

.. doxygenstruct:: spead2::descriptor
   :members:

Streams
-------
At the lowest level, heaps are given to the application via a callback to a
virtual function. While this callback is running, no new packets can be
received from the network socket, so this function needs to complete quickly
to avoid data loss when using UDP. To use this interface, subclass
:cpp:class:`spead2::recv::stream` and implement :cpp:func:`heap_ready` and
optionally override :cpp:func:`stop_received`.

.. doxygenclass:: spead2::recv::stream
   :members: emplace_reader, stop, stop_received, flush, heap_ready

A potentially more convenient interface is
:cpp:class:`spead2::recv::ring_stream\<Ringbuffer>`, which places received
heaps into a fixed-size thread-safe ring buffer. Another thread can then pull
from this ring buffer in a loop. The template parameter selects the ringbuffer
implementation. The default is a good light-weight choice, but if you need to
use :cpp:func:`select`-like functions to wait for data, you can use
:cpp:class:`spead2::ringbuffer\<spead2::recv::live_heap, spead2::semaphore_fd, spead2::semaphore>`.

.. doxygenclass:: spead2::recv::ring_stream

Readers
-------
Reader classes are constructed inside a stream by calling
:cpp:func:`spead2::recv::stream::emplace_reader`.

.. doxygenclass:: spead2::recv::udp_reader
   :members: udp_reader

.. doxygenclass:: spead2::recv::mem_reader
   :members: mem_reader

Memory allocators
-----------------
In addition to the memory allocators described in :ref:`py-memory-allocators`,
new allocators can be created by subclassing :cpp:class:`spead2::memory_allocator`.
For an allocator set on a stream, a pointer to a
:cpp:class:`spead2::recv::packet_header` is passed as a hint to the allocator,
allowing memory to be placed according to information in the packet. Note that
this can be any packet from the heap, so you must not rely on it being the
initial packet.

.. doxygenclass:: spead2::memory_allocator
   :members: allocate, free
