Chunking stream groups
======================
.. cpp:namespace-push:: spead2::recv

While the :doc:`recv-chunk` allows for high-bandwidth streams to be received
with low overhead, it still has a fundamental scaling limitation: each chunk
can only be constructed from a single thread. :dfn:`Chunk stream groups` allow
this overhead to be overcome, although not without caveats.

Each stream is still limited to a single thread. However, a :dfn:`group` of
streams can share the same sequence of chunks, with each stream contributing
a subset of the data in each chunk. Making use of this feature requires
that load balancing is implemented at the network level, using different
destination addresses or ports so that the incoming heaps can be multiplexed
into multiple streams.

As with a single chunk stream, the group keeps a sliding window of chunks and
obtains new ones from an allocation callback. When the window slides forward,
chunks that fall out the back of the window are provided to a ready callback.
Each member stream also has its own sliding window, which can be smaller (but not
larger) than the group's window. When the group's window slides forward, the
streams' windows are adjusted to ensure they still fit within the group's
window. In other words, a stream's window determines how much reordering is
tolerated within a stream, while the group's window determines how out-of-sync
the streams are allowed to become.

When desynchronisation does occur, there is a choice of strategies. The default
strategy is eager but potentially lossy: when the group's window moves forward,
the trailing chunk is marked ready as soon as possible, even if this causes
some stream windows to shrink below their normal size. An alternative strategy
is lossless: when the group's window needs to move forward, it is blocked
until all the member streams have caught up. This latter mode is intended for
use with lossless transports such as TCP. However, if one of the component streams
stops functioning (for example, because it is routed on a network path that is
down) it prevents the entire group from making forward progress.

The general flow (in C++) is

1. Create a :cpp:class:`chunk_stream_group_config`.
2. Create a :cpp:class:`chunk_stream_group`.
3. Use :cpp:func:`chunk_stream_group::emplace_back` to
   create the streams.
4. Add readers to the streams.
5. Process the data.
6. Optionally, call :cpp:func:`chunk_stream_group::stop()`
   (otherwise it will be called on destruction).
7. Destroy the group.

In Python the process is similar, although garbage collection replaces
explicit destruction.

Ringbuffer convenience API
--------------------------
As for standalone chunk streams, there is a simplified API using ringbuffers,
which is also the only API available for Python. A
:cpp:class:`chunk_stream_ring_group` is a group that allocates
data from one ringbuffer and send ready data to another. The description of
:ref:`that api <recv-chunk-ringbuffer>` largely applies here too. The
ringbuffers can be shared between groups.

Caveats
-------
This is an advanced API that sacrifices some user-friendliness for
performance, and thus some care is needed to use it safely.

- It is vital that all the streams can make forward progress independently,
  as otherwise deadlocks can occur. For example, if they share a thread pool,
  the pool must have at least as many threads as streams. It's recommended
  that each stream has its own single-threaded thread pool.
- The streams must all be added to the group before adding any readers to
  the streams. Once a group has received some data, an exception will be thrown
  if one attempts to add a new stream.
- The stream ID associated with each chunk will be the stream ID of one of the
  component streams, but it is undefined which one.
- When the allocate and ready callbacks are invoked, it's not specified which
  stream's batch statistics pointer will be passed.
- Two streams must not write to the same bytes of a chunk (in the payload,
  present array or extra data), as this is undefined behaviour in C++.
- Calling :cpp:func:`~stream::stop` on a member stream will stop the whole
  group.

.. cpp:namespace-pop::
