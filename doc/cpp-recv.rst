Receiving
=========

At the lowest level, heaps are given to the application via a callback to a
virtual function. While this callback is running, no new packets can be
received from the network socket, so this function needs to complete quickly
to avoid data loss when using UDP. To use this interface, subclass
:cpp:class:`spead2::recv::stream` and implement :cpp:func:`heap_ready` and
optionally override :cpp:func:`stop_received`.

There are two heap classes: *live heaps*
(:cpp:class:`spead2::recv::live_heap`) are used for heaps being constructed,
and may be missing data; *frozen heaps* (:cpp:class:`spead2::recv::heap`)
always have all their data. Frozen heaps can be move-constructed from live
heaps, which will typically be done in the callback. A difference from the
Python interface is that incomplete heaps will also be passed to the callback
when they are abandoned. Use
:cpp:func:`spead2::recv::live_heap::is_contiguous` to check whether a live
heap can be frozen.

A potentially more convenient interface is
:cpp:class:`spead2::recv::ring_stream\<T>`, which places received heaps into a
fixed-size thread-safe ring buffer. Another thread can then pull from this
ring buffer in a loop. The template parameter selects a ring buffer
implementation. A light-weight choice using C++11 condition variables is
:cpp:class:`spead2::ringbuffer_cond\<spead2::recv::live_heap>`. A heavier-weight
alternative that can be connected to :cpp:func:`select`-like functions is
:cpp:class:`spead2::ringbuffer_semaphore\<spead2::recv::live_heap>`.
