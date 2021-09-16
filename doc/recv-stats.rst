Receiver stream statistics
==========================

A receive stream can be queried for statistics about the packets and heaps of
the stream. Note that while the statistics below are expected to
be stable except where otherwise noted, their exact interpretation in edge
cases is subject to change as the implementation evolves. It is intended for
instrumentation, rather than for driving application logic.

Each time the statistics are queried, an internally consistent view is
returned. However, it is not synchronised with other aspects of
the stream. For example, it's theoretically possible to retrieve 5 heaps from
the stream iterator, then find that the ``heaps`` statistic is (briefly)
4. Querying the statistics is somewhat expensive, so if multiple statistics
are needed, it is advisable to assign the result to a local variable first.

Some readers process packets in batches, and the statistics are only updated
after a whole batch is added. This can be particularly noticeable if the
ringbuffer fills up and blocks the reader, as this prevents the batch from
completing and so heaps that have already been received by user code might
not be reflected in the statistics.

Statistics can be accessed in several ways:

1. A map/dictionary-like interface, by name. This includes iteration with
   standard Python or C++ iterator conventions.
2. A list-like interface, by index. This is more efficient, so may be useful
   if statistics are being accessed very frequently, but less convenient. To
   determine the index for a particular statistic, use
   :py:meth:`.StreamConfig.get_stat_index` (Python) or
   :cpp:func:`spead2::recv::stream_config::get_stat_index` (C++).
3. By attribute/field access. This is for backwards compatibility and is only
   available for the core statistics. New code should prefer the other
   interfaces.

Core statistics
---------------

heaps
   Total number of heaps put into the stream. This includes incomplete heaps,
   and complete heaps that were received but did not make it into the
   ringbuffer before :py:meth:`~spead2.recv.Stream.stop` was called. It
   excludes the heap that contained the stop item.

incomplete_heaps_evicted
   Number of incomplete heaps that were evicted from the buffer to make room
   for new data.

incomplete_heaps_flushed
   Number of incomplete heaps that were still in the buffer when the stream
   stopped.

packets
   Total number of packets received, including the one containing the stop
   item.

batches
   Number of batches of packets. Some readers are able to take multiple packets
   from the network in one go, and each time this forms a batch.

worker_blocked
   Number of times a worker thread was blocked because the ringbuffer was full.
   If this is non-zero, it indicates that the stream is not being read fast
   enough, or that the `heaps` constructor parameter to the
   :class:`~.RingStreamConfig` needs to be
   increased to buffer sudden bursts.

   In C++ this statistic is always present, but is only used by
   :cpp:class:`spead2::recv::ring_stream`.

max_batch
   Maximum number of packets received as a unit. This is only applicable to
   readers that support fetching a batch of packets from the source.

single_packet_heaps
   Number of heaps that were entirely contained in a single packet. These
   take a slightly faster path as it is not necessary to reassemble them.

search_dist
   Number of hash table entries searched to find the heaps associated with
   packets. This is intended for debugging/profiling spead2 and **may be
   removed without notice**.

Chunk receiver statistics
-------------------------

These statistics are only present when using a :doc:`chunking receiver <recv-chunk>`.

too_old_heaps
    Heaps for which the chunk placement function returned a non-negative chunk
    ID, but one which was too old to be accepted (behind the moving window).

rejected_heaps
    Heaps for which the chunk placement function returned a negative chunk ID
    to indicate that the heap should be discarded.

.. _custom-stats:

Custom statistics
-----------------

It may be convenient for user code to collect additional statistics to be made
available through the existing statistics framework. The framework takes care
of thread-safe transfer of statistics from the worker threads that run the
networking to the thread that is querying the statistics. However, this means
that these custom statistics can *only* be safely updated from these worker
threads. Some examples of places where it is safe to do so:

1. In the ``heap_ready`` virtual function.
2. In a :ref:`custom memory allocator <memory-allocators>` (when called to
   allocate memory for a heap).
3. In a :ref:`custom memory scatter <custom-memory-scatter>` function.
4. In a chunk placement callback (see :doc:`recv-chunk`).

All but the last are currently available in the C++ API only. They should use
:cpp:member:`spead2::recv::stream::batch_stats`. For the chunk placement
callback, a pointer to the batch statistics is available in the
:cpp:struct:`spead2::recv::chunk_place_data` structure.

As the name implies, this provides access only to statistics collected for a
batch of packets received at the same time. At the end of the batch, the
long-term statistics for the stream are updated from these batch statistics.
The manner in which this update occurs depends on the *mode* of the statistic,
which is one of the following:

counter
    A count of events. The batch value is added to the long-term value.
maximum
    A high water mark. The long-term value is set to the maximum of the
    previous value and the batch value.

The mode is set when registering the statistic with the stream config
(:py:meth:`.StreamConfig.add_stat` or
:cpp:func:`spead2::recv::stream_config::add_stat`).

Registration also returns the "index" of the statistic, which is used when
accessing the batch statistics array. If many statistics are being registered,
it may be inconvenient to keep track of all their indices. The index is
guaranteed to increase by one with each registration, so one can instead
record just the first index, and then compute other indices from it as needed.

Since all statistics (custom and core) share a single namespace, it is
recommended that you prefix your custom statistics with a package name and a
dot (``mypackage.mystatistic``) to ensure that they do not conflict with
future statistics added by spead2. It's also recommended to stick to printable
ASCII for maximum compatibility across language bindings.
