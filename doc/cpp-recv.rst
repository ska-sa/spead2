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
:cpp:class:`spead2::recv::ring_stream\<T>`, which places received heaps into a
fixed-size thread-safe ring buffer. Another thread can then pull from this
ring buffer in a loop. The template parameter selects a ring buffer
implementation. A light-weight choice using C++11 condition variables is
:cpp:class:`spead2::ringbuffer_cond\<spead2::recv::live_heap>`. A heavier-weight
alternative that can be connected to :cpp:func:`select`-like functions is
:cpp:class:`spead2::ringbuffer_semaphore\<spead2::recv::live_heap>`.

.. doxygenclass:: spead2::recv::ring_stream

Readers
-------
Reader classes are constructed inside a stream by calling
:cpp:func:`spead2::recv::stream::emplace_reader`.

.. doxygenclass:: spead2::recv::udp_reader
   :members: udp_reader

.. doxygenclass:: spead2::recv::mem_reader
   :members: mem_reader
