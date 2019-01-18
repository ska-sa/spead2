Receiving
=========

Heaps
-----
Unlike the Python bindings, the C++ bindings expose three heap types: *live heaps*
(:cpp:class:`spead2::recv::live_heap`) are used for heaps being constructed,
and may be missing data; *frozen heaps* (:cpp:class:`spead2::recv::heap`)
always have all their data; and
*incomplete heaps* (:cpp:class:`spead2::recv::incomplete_heap`) are frozen
heaps that are missing data. Frozen heaps can be move-constructed from live
heaps, which will typically be done in the callback.

.. doxygenclass:: spead2::recv::live_heap
   :members: is_complete,is_contiguous,is_end_of_stream,get_cnt,get_bug_compat

.. doxygenclass:: spead2::recv::heap
   :members:

.. doxygenclass:: spead2::recv::incomplete_heap
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

Note that some public functions are incorrectly listed as protected below due
to limitations of the documentation tools.

.. doxygenclass:: spead2::recv::stream
   :members:

.. doxygenstruct:: spead2::recv::stream_stats
   :members:

A potentially more convenient interface is
:cpp:class:`spead2::recv::ring_stream\<Ringbuffer>`, which places received
heaps into a fixed-size thread-safe ring buffer. Another thread can then pull
from this ring buffer in a loop. The template parameter selects the ringbuffer
implementation. The default is a good light-weight choice, but if you need to
use :cpp:func:`select`-like functions to wait for data, you can use
:cpp:class:`spead2::ringbuffer\<spead2::recv::live_heap, spead2::semaphore_fd, spead2::semaphore>`.

.. doxygenclass:: spead2::recv::ring_stream
   :members: ring_stream, pop, try_pop, pop_live, try_pop_live

Readers
-------
Reader classes are constructed inside a stream by calling
:cpp:func:`spead2::recv::stream::emplace_reader`.

.. doxygenclass:: spead2::recv::udp_reader
   :members: udp_reader

.. doxygenclass:: spead2::recv::tcp_reader
   :members: tcp_reader

.. doxygenclass:: spead2::recv::inproc_reader
   :members: inproc_reader

.. doxygenclass:: spead2::recv::mem_reader
   :members: mem_reader

.. doxygenclass:: spead2::recv::udp_pcap_file_reader
   :members: udp_pcap_file_reader

Memory allocators
-----------------
In addition to the memory allocators described in :ref:`py-memory-allocators`,
new allocators can be created by subclassing :cpp:class:`spead2::memory_allocator`.
For an allocator set on a stream, a pointer to a
:cpp:class:`spead2::recv::packet_header` is passed as a hint to the allocator,
allowing memory to be placed according to information in the packet. Note that
for unreliable transport this could be any packet from the heap, and you should
not rely on it being the initial packet.

.. doxygenclass:: spead2::memory_allocator
   :members: allocate, free


Custom memory scatter
---------------------------
In specialised high-bandwidth cases, the overhead of assembling heaps in
temporary storage before scattering the data into other arrangements can be
very high. It is possible (since 1.11) to take complete control over the
transfer of the payload of the SPEAD packets. Before embarking on such an
approach, be sure you have a good understanding of the SPEAD protocol,
particularly packets, heaps, item pointers and payload.

In the simplest case, each heap needs to be written to some special or
pre-allocated storage, but in a contiguous fashion. In this case it is
sufficient to provide a custom allocator (see above), which will return a
pointer to the target storage.

In more complex cases, the contents of each heap, or even each packet, needs
to be scattered to discontiguous storage areas. In this case, one can
additionally override the memory copy function with
:cpp:func:`~spead2::recv::stream_base::set_memcpy` and providing a
:cpp:type:`~spead2::recv::packet_memcpy_function`.

.. doxygentypedef:: spead2::recv::packet_memcpy_function

It takes a pointer to the start of the heap's allocation (as returned by the
allocator) and the packet metadata. The default implementation is equivalent
to the following:

.. code-block:: c++

    void copy(const spead2::memory_allocator::pointer &allocation, const packet_header &packet)
    {
        memcpy(allocation.get() + packet.payload_offset, packet.payload, packet.payload_length);
    }

Note that when providing your own memory copy and allocator, you don't
necessarily need the allocator to actually return a pointer to payload memory.
It could, for example, populate a structure that guides the copy, and return a
pointer to that; or it could return a null pointer. There are some caveats
though:

1. If the sender doesn't provide the heap length item, then spead2 may need to
   make multiple allocations of increasing size as the heap grows, and each
   time it will copy (with standard memcpy, rather than your custom one) the
   old content to the new. Assuming you aren't expecting such packets, you can
   reject them using
   :cpp:func:`~spead2::recv::stream_base::set_allow_unsized_heaps`.

2. :cpp:func:`spead2::recv::heap_base::get_items` constructs pointers to the items
   on the assumption of the default memcpy function, so if your replacement
   doesn't copy things to the same place, you obviously won't be able to use
   those pointers. Note that :cpp:func:`~spead2::recv::heap::get_descriptors`
   will also not be usable.
