Synchronisation in chunk stream groups
======================================
.. cpp:namespace-push:: spead2::recv

For chunk stream groups to achieve the goal of allowing multi-core scaling, it
is necessary to minimise locking. The implementation achieves this by avoiding
any packet- or heap-granularity locking, and performing locking only at chunk
granularity. Chunks are assumed to be large enough that this minimises total
overhead, although it should be noted that these locks are expected to be
highly contended and there may be further work possible to reduce the
overheads.

To avoid the need for heap-level locking, each member stream has its own
sliding window with pointers to the chunks, so that heaps which fall inside an
existing chunk can be serviced without locking. However, this causes a problem
when flushing chunks from the group's window: a stream might still be writing
to the chunk at the time. Additionally, it might not be possible to allocate a
new chunk until an old chunk is flushed e.g., if there is a fixed pool of
chunks rather than dynamic allocation.

Each chunk has a reference count, indicating the number of streams that still
have the chunk in their window. This reference count is non-atomic since it is
protected by the group's mutex. When the group wishes to evict a chunk, it
first needs to wait for the reference count of the head chunk to drop to zero.
It needs a way to be notified that it should try again, which is provided by a
condition variable. Using a condition variable (rather than, say, replacing
the simple reference count with a semaphore) allows the group mutex to be
dropped while waiting, which prevents the deadlocks that might otherwise occur
if the mutex was held while waiting and another stream was attemping to lock
the group mutex to make forward progress.

In lossless eviction mode, this is all that is needed, although it is
non-trivial to see that this won't deadlock with all the streams sitting in
the wait loop waiting for other streams to make forward progress. That this
cannot happen is due to the requirement that the stream's window cannot be
larger than the group's. Consider the active call to
:cpp:func:`chunk_stream_group::get_chunk` with the smallest chunk ID. That
stream is guaranteed to have already readied any chunk due to be evicted from
the group, and the same is true of any other stream that is waiting in
:cpp:func:`~chunk_stream_group::get_chunk`, and so forward progress depends
only on streams that are not blocked in
:cpp:func:`~chunk_stream_group::get_chunk`.

In lossy eviction mode, we need to make sure that such streams make forward
progress even if no new packets arrive on them. This is achieved by posting an
asynchronous callback to all streams requesting them to flush out chunks that
are now too old.

.. cpp:namespace-pop::
