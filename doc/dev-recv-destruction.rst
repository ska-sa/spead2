Destruction of receive streams
==============================

.. cpp:namespace:: spead2::recv

The asynchronous and parallel nature of spead2 makes destroying a receive
stream a tricky operation: there may be pending asio completion handlers that
will try to push packets into the stream, leading to a race condition. While
asio guarantees that closing a socket will cancel any pending asynchronous
operations on that socket, this doesn't account for cases where the operation
has already completed but the completion handler is either pending or is
currently running.

Up to version 3.11, this was handled by a shutdown protocol
between :cpp:class:`stream` and
:cpp:class:`reader`. The reader was required to notify the
stream when it had completely shut down, and
:cpp:func:`stream::stop` would block until all readers had
performed this notification (via a semaphore). This protocol was complicated,
and it relied on the reader being able to make forward progress while the
thread calling :cpp:func:`stream::stop` was blocked.

Newer versions take a different approach based on shared pointers. The ideal
case would be to have the whole stream always managed by a shared pointer, so
that a completion handler that interfaces with the stream could keep a copy of
the shared pointer and thus keep it alive as long as needed. However, that is
not possible to do in a backwards-compatible way. Instead, a minimal set of
fields is placed inside a shared pointer, namely:

- The ``queue_mutex``
- A flag indicating whether the stream has stopped.

For convenience, the flag is encoded as a pointer, which holds either a
pointer to the stream (if not stopped) or a null pointer (if stopped). Each
completion handler holds a shared reference to this structure. When it wishes
to access the stream, it should:

1. Lock the mutex.
2. Get the pointer back to the stream from the shared structure, aborting if
   it gets a null pointer.
3. Manipulate the stream.
4. Drop the mutex.

This prevents use-after-free errors because the stream cannot be destroyed
without first stopping, and stopping locks the mutex. Hence, the stream cannot
disappear asynchronously during step 3. Note that it can, however, stop
during step 3 if the completion handler causes it to stop. Some protection is
added for this: :cpp:func:`stream::add_packet_handler::add_packet` will not
immediately stop the stream if a stop packet is received; instead, it will
stop it when the :cpp:class:`stream::add_packet_handler` is destroyed.

Using shared pointers in this way can add overhead because atomically
incrementing and decrementing reference counts can be expensive, particularly
if it causes cache line migrations between processor cores. To minimise
reference count manipulation, the :cpp:class:`reader` class
encapsulates this workflow in its
:cpp:func:`~reader::bind_handler` member function, which
provides the facilities to move the shared pointer along a linear chain of
completion handlers so that the reference count does not need to be
adjusted.

Readers are only destroyed when the stream is destroyed. This ensures that the
reader's destructor is called from a user's thread (which in Python bindings,
will hold the GIL). To handle more immediate cleanup when a stream is stopped,
readers may override :cpp:func:`reader::stop`.
