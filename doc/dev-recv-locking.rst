Locking and asio for receive
============================

.. cpp:namespace:: spead2::recv

Spead2 uses a somewhat unusual combination of locking with asynchronous I/O.
It causes a few problems and ideally should be redesigned one day. First,
let's introduce some terminology: the user writes code which runs on a thread,
which we'll call the "user thread" (there may be multiple user threads, but
for most functions it's only safe for one user thread at a time to interact
with a stream). The threads in a :py:class:`.ThreadPool`, or more generally
threads running :cpp:func:`boost::asio::io_service::run` (or equivalents) are
"worker threads".

The original sin that leads to many complications is
the back-pressure mechanism for receiving: if a ring-buffer is full, then
pushing a heap (or chunk) to it simply blocks the worker thread, rather than
signalling to readers that they should stop listening for data until space
becomes available. Blocking a worker thread is generally a bad thing to do in
asynchronous programming, and if not handled carefully can lead to deadlocks.
Even when it is safe, it can lead to inefficiencies since the blocked thread
is sitting idle when there could be other work for it. This is one reason that
sharing thread pools between streams is not recommended (another is cache
locality). Fixing this would require major and backwards-incompatible redesign
to allow for control-flow signalling.

Locking is needed for a few reasons:

- The user thread and a worker thread may need to access the same data.
  Version 3 reduced the number of places this can happen by making most of the
  configuration immutable, but it is still needed to stop the stream and to
  access statistics.
- In a stream with multiple readers and multiple worker threads, it is
  possible for multiple worker threads to need access to the stream internals
  concurrently.

Early versions of spead2 solved these problems using strands_, where functions
invoked by the user thread would post work to the strand and use futures to
return the result to the user thread. This lead to many issues with deadlocks,
and debugging was difficult because this control flow was not apparent in the
call stack. It may be worth revisiting now that there are fewer places where
the user thread needs to interact with the stream internals, but it will be
necessary to compare the performance to the locking approach.

.. _strands: https://www.boost.org/doc/libs/1_83_0/doc/html/boost_asio/overview/core/strands.html

Batching
--------
Even in the absence of contention, locking can be expensive, and we found that
taking and releasing a lock for every packet had a significant cost. The
design was thus changed to ensure that multiple packets can be handled with a
single lock. This complements APIs such as :func:`recvmmsg` that allow
multiple available packets to be retrieved at once.

This batching approach is realised by the
:cpp:class:`spead2::recv::stream::add_packet_state` class. Constructing the
class takes a lock on the stream, and the destructor releases it. This class
also holds local statistics for the batch, which are used to update the
stream-wide statistics at the end of the batch.

Unfortunately, pushing completed heaps to the user is done with this lock
held, which means that not only is the worker thread blocked if the ringbuffer
is full, but any other thread (including a user thread) that needs the lock
will also be blocked.

Stopping
--------
There are four circumstances under which a receive stream can stop:

1. A stream control heap is received from the network.

2. A transport-level event occurs, such as a connection being
   closed by the remote end.

3. The user calls :cpp:func:`stream::stop`.

4. The user destroys the stream (which implicitly calls
   :cpp:func:`stream::stop`).

The first two are referred to as "network stops" and the latter two as "user
stops". Both cases involve call :cpp:func:`stream::stop_received`, but
only user stops invoke :cpp:func:`stream::stop`.

A fundamental difference between the two cases is that for network stops, the
user is generally waiting for data from the stream, and so one can assume that
the ringbuffer will generally empty out in finite time. What's more, the user
may wish to actually receive all the data that was transmitted prior to the
network stop. With user stops, the user is generally not consuming from the
ringbuffer, and it must be possible to stop the stream even if the ringbuffer
is full, even if this means losing data that is still arriving from the
network.

To handle user stops correctly, stream classes whose
:cpp:func:`stream_base::heap_ready` function potentially blocks must override
:cpp:func:`~stream::stop` to unblock it. Classes that use ringbuffers
(:cpp:class:`ring_stream`, :cpp:class:`chunk_ring_stream` etc.) do so by
stopping the ringbuffer *before* calling the base class implementation. This
causes any blocked (and future) attempts to push data into the ringbuffer to
immediately fail with an exception. This does mean that some data that was
received is dropped. On the other hand, network stops do *not* immediately
stop the ringbuffer, and allow any data still in the stream to be flushed.
This does mean that if there is no consumer for the ringbuffer, the worker
thread could be blocked until the user stop (or resumes consuming from the
ringbuffer).
