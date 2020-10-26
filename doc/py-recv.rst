Receiving
---------
The classes associated with receiving are in the :py:mod:`spead2.recv`
package. A *stream* represents a logical stream, in that packets with
the same heap ID are assumed to belong to the same heap. A stream can have
multiple physical transports.

Streams yield *heaps*, which are the basic units of data transfer and contain
both item descriptors and item values. While it is possible to directly
inspect heaps, this is not recommended or supported. Instead, heaps are
normally passed to :py:meth:`spead2.ItemGroup.update`.

.. py:class:: spead2.recv.Heap

   .. py:attribute:: cnt

      Heap identifier (read-only)

   .. py:attribute:: flavour

      SPEAD flavour used to encode the heap (see :ref:`py-flavour`)

   .. py:function:: is_start_of_stream()

      Returns true if the packet contains a stream start control item.

   .. py:function:: is_end_of_stream()

      Returns true if the packet contains a stream stop control item.

.. note:: Malformed packets (such as an unsupported SPEAD version, or
  inconsistent heap lengths) are dropped, with a log message. However,
  errors in interpreting a fully assembled heap (such as invalid/unsupported
  formats, data of the wrong size and so on) are reported as
  :py:exc:`ValueError` exceptions. Robust code should thus be prepared to
  catch exceptions from heap processing.

Configuration
^^^^^^^^^^^^^
Once a stream is constructed, the configuration cannot be changed. The configuration is
captured in two classes, :py:class:`~spead2.recv.StreamConfig` and
:py:class:`~spead2.recv.RingStreamConfig`. The split is a reflection of the C++
API and not particularly relevant in Python. The configuration options can
either be passed to the constructors (as keyword arguments) or set as
properties after construction.

.. py:class:: spead2.recv.StreamConfig(**kwargs)

   :param int max_heaps:
     The number of partial heaps that can be live at one time.
     This affects how intermingled heaps can be (due to out-of-order packet
     delivery) before heaps get dropped.
   :param int bug_compat:
     Bug compatibility flags (see :ref:`py-flavour`)
   :param int memcpy:
     Set the method used to copy data from the network to the heap. The
     default is :py:const:`~spead2.MEMCPY_STD`. This can be changed to
     :py:const:`~spead2.MEMCPY_NONTEMPORAL`, which writes to the destination with a
     non-temporal cache hint (if SSE2 is enabled at compile time). This can
     improve performance with large heaps if the data is not going to be used
     immediately, by reducing cache pollution. Be careful when benchmarking:
     receiving heaps will generally appear faster, but it can slow down
     subsequent processing of the heap because it will not be cached.
   :param memory_allocator:
     Set the memory allocator for a stream. See
     :ref:`py-memory-allocators` for details.
   :type memory_allocator: :py:class:`spead2.MemoryAllocator`
   :param bool stop_on_stop_item:
     By default, a heap containing a stream control stop item will terminate
     the stream (and that heap is discarded). In some cases it is useful to
     keep the stream object alive and ready to receive a following stream.
     Setting this attribute to ``False`` will disable this special
     treatment. Such heaps can then be detected with
     :meth:`~spead2.recv.Heap.is_end_of_stream`.
   :param bool allow_unsized_heaps:
     By default, spead2 caters for heaps without a `HEAP_LEN` item, and will
     dynamically extend the memory allocation as data arrives. However, this
     can be expensive, and ideally senders should include this item. Setting
     this attribute to ``False`` will cause packets without this item to be
     rejected.
   :param bool allow_out_of_order:
     Whether to allow packets within a heap to be received out-of-order. See
     :ref:`py-packet-ordering` for details.
   :raises ValueError: if `max_heaps` is zero.

.. py:class:: spead2.recv.RingStreamConfig(**kwargs)

   :param int heaps: The capacity of the ring buffer between the network
     threads and the consumer. Increasing this may reduce lock contention at
     the cost of more memory usage.
   :param bool contiguous_only: If set to ``False``, incomplete heaps will be
     included in the stream as instances of :py:class:`.IncompleteHeap`. By
     default they are discarded. See :ref:`py-incomplete-heaps` for details.
   :param bool incomplete_keep_payload_ranges: If set to ``True``, it is
     possible to retrieve information about which parts of the payload arrived
     in incomplete heaps, using :py:meth:`.IncompleteHeap.payload_ranges`.
   :raises ValueError: if `ring_heaps` is zero.

Blocking receive
^^^^^^^^^^^^^^^^
To do blocking receive, create a :py:class:`spead2.recv.Stream`, and add
transports to it with :py:meth:`~spead2.recv.Stream.add_buffer_reader`,
:py:meth:`~spead2.recv.Stream.add_udp_reader`,
:py:meth:`~spead2.recv.Stream.add_tcp_reader` or
:py:meth:`~spead2.recv.Stream.add_udp_pcap_file_reader`. Then either iterate over
it, or repeatedly call :py:meth:`~spead2.recv.Stream.get`.

.. py:class:: spead2.recv.Stream(thread_pool, stream_config=StreamConfig(), ring_config=RingStreamConfig())

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param config: Stream configuration
   :type config: :py:class:`spead2.recv.StreamConfig`
   :param ring_config: Ringbuffer configuration
   :type ring_config: :py:class:`spead2.recv.RingStreamConfig`

   .. py:attribute:: config

      Stream configuration passed to the constructor (read-only)

   .. py:attribute:: ring_config

      Ringbuffer configuration passed to the constructor (read-only)

   .. py:method:: add_buffer_reader(buffer)

      Feed data from an object implementing the buffer protocol.

   .. py:method:: add_udp_reader(port, max_size=DEFAULT_UDP_MAX_SIZE, buffer_size=DEFAULT_UDP_BUFFER_SIZE, bind_hostname='', socket=None)

      Feed data from a UDP port.

      :param int port: UDP port number
      :param int max_size: Largest packet size that will be accepted.
      :param int buffer_size: Kernel socket buffer size. If this is 0, the OS
        default is used. If a buffer this large cannot be allocated, a warning
        will be logged, but there will not be an error.
      :param str bind_hostname: If specified, the socket will be bound to the
        first IP address found by resolving the given hostname. If this is a
        multicast group, then it will also subscribe to this multicast group.

   .. py:method:: add_udp_reader(multicast_group, port, max_size=DEFAULT_UDP_MAX_SIZE, buffer_size=DEFAULT_UDP_BUFFER_SIZE, interface_address)
      :noindex:

      Feed data from a UDP port (IPv4 only). This is intended for use with
      multicast, but it will also accept a unicast address as long as it is the
      same as the interface address.

      :param str multicast_group: Hostname/IP address of the multicast group to subscribe to
      :param int port: UDP port number
      :param int max_size: Largest packet size that will be accepted.
      :param int buffer_size: Kernel socket buffer size. If this is 0, the OS
        default is used. If a buffer this large cannot be allocated, a warning
        will be logged, but there will not be an error.
      :param str interface_address: Hostname/IP address of the interface which
        will be subscribed, or the empty string to let the OS decide.

   .. py:method:: add_udp_reader(multicast_group, port, max_size=DEFAULT_UDP_MAX_SIZE, buffer_size=DEFAULT_UDP_BUFFER_SIZE, interface_index)
      :noindex:

      Feed data from a UDP port with multicast (IPv6 only).

      :param str multicast_group: Hostname/IP address of the multicast group to subscribe to
      :param int port: UDP port number
      :param int max_size: Largest packet size that will be accepted.
      :param int buffer_size: Kernel socket buffer size. If this is 0, the OS
        default is used. If a buffer this large cannot be allocated, a warning
        will be logged, but there will not be an error.
      :param str interface_index: Index of the interface which will be
        subscribed, or 0 to let the OS decide.

   .. py:method:: add_tcp_reader(port, max_size=DEFAULT_TCP_MAX_SIZE, buffer_size=DEFAULT_TCP_BUFFER_SIZE, bind_hostname='')

      Receive data over TCP/IP. This will listen for a single incoming
      connection, after which no new connections will be accepted. When the
      connection is closed, the stream is stopped.

      :param int port: TCP port number
      :param int max_size: Largest packet size that will be accepted.
      :param int buffer_size: Kernel socket buffer size. If this is 0, the OS
        default is used. If a buffer this large cannot be allocated, a warning
        will be logged, but there will not be an error.
      :param str bind_hostname: If specified, the socket will be bound to the
        first IP address found by resolving the given hostname.

   .. py:method:: add_tcp_reader(acceptor, max_size=DEFAULT_TCP_MAX_SIZE)
      :noindex:

      Receive data over TCP/IP. This is similar to the previous overload, but
      takes a user-provided socket, which must already be listening for
      connections. It duplicates the acceptor socket, so the original can be
      closed immediately.

      :param socket.socket acceptor: Listening socket
      :param int max_size: Largest packet size that will be accepted.

   .. py:method:: add_udp_pcap_file_reader(filename)

      Feed data from a pcap file (for example, captured with :program:`tcpdump`
      or :ref:`mcdump`). This is only available if libpcap development files
      were found at compile time.

   .. py:method:: add_inproc_reader(queue)

      Feed data from an in-process queue. Refer to :doc:`py-inproc` for details.

   .. py:method:: get()

      Returns the next heap, blocking if necessary. If the stream has been
      stopped, either by calling :py:meth:`stop` or by receiving a stream
      control packet, it raises :py:exc:`spead2.Stopped`. However, heap that
      were already queued when the stream was stopped are returned first.

      A stream can also be iterated over to yield all heaps.

   .. py:method:: get_nowait()

      Like :py:meth:`get`, but if there is no heap available it raises
      :py:exc:`spead2.Empty`.

   .. py:method:: stop()

      Shut down the stream and close all associated sockets. It is not
      possible to restart a stream once it has been stopped; instead, create a
      new stream.

   .. py:attribute:: fd

      The read end of a pipe to which a byte is written when a heap is
      received. **Do not read from this pipe.** It is used for integration
      with asynchronous I/O frameworks (see below).

   .. py:attribute:: stats

      Statistics_ about the stream.

   .. py:attribute:: ringbuffer

      The internal ringbuffer of the stream (see Statistics_).

Asynchronous receive
^^^^^^^^^^^^^^^^^^^^
Asynchronous I/O is supported through Python's :py:mod:`asyncio` module. It can
be combined with other asynchronous I/O frameworks like twisted_ and Tornado_.

.. py:class:: spead2.recv.asyncio.Stream(*args, **kwargs)

   See :py:class:`spead2.recv.Stream` (the base class) for other constructor
   arguments.

   .. py:method:: get()

      Coroutine that yields the next heap, or raises :py:exc:`spead2.Stopped`
      once the stream has been stopped and there is no more data. It is safe
      to have multiple in-flight calls, which will be satisfied in the order
      they were made.

.. _twisted: https://twistedmatrix.com/trac/
.. _tornado: http://www.tornadoweb.org/en/stable/

The stream is also asynchronously iterable, i.e., can be used in an ``async
for`` loop to iterate over the heaps.

.. _py-packet-ordering:

Packet ordering
^^^^^^^^^^^^^^^
SPEAD is typically carried over UDP, and by its nature, UDP allows packets to
be reordered. Packets may also arrive interleaved if they are produced by
multiple senders. We consider two sorts of packet ordering issues:

1. Re-ordering within a heap. By default, spead2 assumes that all the packets
   that form a heap will arrive in order, and discards any packet that does
   not have the expected payload offset. In most networks this is a safe
   assumption provided that all the packets originate from the same sender (IP
   address and port number) and have the same destination.

   If this assumption is not appropriate, it can be changed with the
   :py:attr:`allow_out_of_order` attribute of
   :py:class:`spead2.recv.StreamConfig`. This has minimal impact when packets
   do in fact arrive in order, but reassembling arbitrarily ordered packets
   can be expensive. Allowing for out-of-order arrival also makes handling
   lost packets more expensive (because one must cater for them arriving
   later), which can lead to a feedback loop as this more expensive processing
   can lead to further packet loss.

2. Interleaving of packets from different heaps. This is always supported, but
   to a bounded degree so that lost packets don't lead to heaps being kept
   around indefinitely in the hope that the packet may arrive. The
   :py:attr:`max_heaps` attribute of :py:class:`spead2.recv.StreamConfig`
   determines the amount of overlap allowed: once a packet in heap :math:`n`
   is observed, it is assumed that heap :math:`n - \text{max_heaps}` is
   complete. When there are many producers it will likely to be necessary to
   increase this value. Larger values increase the memory usage for partial
   heaps, and have a small performance impact.

.. _py-memory-allocators:

Memory allocators
^^^^^^^^^^^^^^^^^
To allow for performance tuning, it is possible to use an alternative memory
allocator for heap payloads. A few allocator classes are provided; new classes
must currently be written in C++. The default (which is also the base class
for all allocators) is :py:class:`spead2.MemoryAllocator`, which has no
constructor arguments or methods. An alternative is
:py:class:`spead2.MmapAllocator`.

.. py:class:: spead2.MmapAllocator(flags=0)

    An allocator using :manpage:`mmap(2)`. This may be slightly faster for large
    allocations, and allows setting custom mmap flags. This is mainly intended
    for use with the C++ API, but is exposed to Python as well.

    :param int flags:
        Extra flags to pass to :manpage:`mmap(2)`. Finding the numeric values
        for OS-specific flags is left as a problem for the user.

The most important custom allocator is :py:class:`spead2.MemoryPool`. It allocates
from a pool, rather than directly from the system. This can lead to
significant performance improvements when the allocations are large enough
that the C library allocator does not recycle the memory itself, but instead
requests memory from the kernel.

A memory pool has a range of sizes that it will handle from its pool, by
allocating the upper bound size. Thus, setting too wide a range will waste
memory, while setting too narrow a range will prevent the memory pool from
being used at all. A memory pool is best suited for cases where the heaps are
all roughly the same size.

A memory pool can optionally use a background task (scheduled onto a thread
pool) to replenish the pool when it gets low. This is useful when heaps are
being captured and stored indefinitely rather than processed and released.

.. py:class:: spead2.MemoryPool(thread_pool, lower, upper, max_free, initial, low_water, allocator=None)

   Constructor. One can omit `thread_pool` and `low_water` to skip the
   background refilling.

   :param ThreadPool thread_pool: thread pool used for
     refilling the memory pool
   :param int lower: Minimum allocation size to handle with the pool
   :param int upper: Size of allocations to make
   :param int max_free: Maximum number of allocations held in the pool
   :param int initial: Number of allocations to put in the free pool
     initially.
   :param int low_water: When fewer than this many buffers remain, the
     background task will be started and allocate new memory until `initial`
     buffers are available.
   :param MemoryAllocator allocator: Underlying memory allocator

   .. py:attribute:: warn_on_empty

      Whether to issue a warning if the memory pool becomes empty and needs to
      allocate new memory on request. It defaults to true.

.. _py-incomplete-heaps:

Incomplete Heaps
^^^^^^^^^^^^^^^^
By default, an incomplete heap (one for which some but not all of the packets
were received) is simply dropped and a warning is printed. Advanced users
might need finer control, such as recording metrics about the number of these
heaps. To do so, set `contiguous_only` to ``False`` in the
:py:class:`~spead2.recv.RingStreamConfig`. The stream will then yield
instances of :py:class:`.IncompleteHeap`.

.. py:class:: spead2.recv.IncompleteHeap

   .. py:attribute:: cnt

      Heap identifier (read-only)

   .. py:attribute:: flavour

      SPEAD flavour used to encode the heap (see :ref:`py-flavour`)

   .. py:attribute:: heap_length

      The expected number of bytes of payload (-1 if unknown)

   .. py:attribute:: received_length

      The number of bytes of payload that were actually received

   .. py:attribute:: payload_ranges

      A list of pairs of heap offsets. Each pair is a range of bytes that was
      received. This is only non-empty if `incomplete_keep_payload_ranges` was
      set in the :py:class:`~spead2.recv.RingStreamConfig`; otherwise the
      information is dropped to save memory.

      When using this, you should also set `allow_out_of_order` to ``True`` in
      the :py:class:`~spead2.recv.StreamConfig`, as otherwise any data after
      the first lost packet is discarded.

   .. py:function:: is_start_of_stream()

      Returns true if the packet contains a stream start control item.

   .. py:function:: is_end_of_stream()

      Returns true if the packet contains a stream stop control item.


.. Statistics:

Statistics
^^^^^^^^^^
The :py:attr:`~spead2.recv.Stream.stats` property of a stream contains
statistics about the stream. Note that while the fields below are expected to
be stable except where otherwise noted, their exact interpretation in edge
cases is subject to change as the implementation evolves. It is intended for
instrumentation, rather than for driving application logic.

Each time the property is accessed, an internally consistent view of the
statistics is returned. However, it is not synchronised with other aspects of
the stream. For example, it's theoretically possible to retrieve 5 heaps from
the stream iterator, then find that :py:attr:`.StreamStats.heaps` is (briefly)
4.

Some readers process packets in batches, and the statistics are only updated
after a whole batch is added. This can be particularly noticeable if the
ringbuffer fills up and blocks the reader, as this prevents the batch from
completing and so heaps that have already been received by Python code might
not be reflected in the statistics.

.. py:class:: spead2.recv.StreamStats

   .. py:attribute:: heaps

   Total number of heaps put into the stream. This includes incomplete heaps,
   and complete heaps that were received but did not make it into the
   ringbuffer before :py:meth:`~spead2.recv.Stream.stop` was called. It
   excludes the heap that contained the stop item.

   .. py:attribute:: incomplete_heaps_evicted

   Number of incomplete heaps that were evicted from the buffer to make room
   for new data.

   .. py:attribute:: incomplete_heaps_flushed

   Number of incomplete heaps that were still in the buffer when the stream
   stopped.

   .. py:attribute:: packets

   Total number of packets received, including the one containing the stop
   item.

   .. py:attribute:: batches

   Number of batches of packets. Some readers are able to take multiple packets
   from the network in one go, and each time this forms a batch.

   .. py:attribute:: worker_blocked

   Number of times a worker thread was blocked because the ringbuffer was full.
   If this is non-zero, it indicates that the stream is not being read fast
   enough, or that the `ring_heaps` constructor parameter needs to be
   increased to buffer sudden bursts.

   .. py:attribute:: max_batch

   Maximum number of packets received as a unit. This is only applicable to
   readers that support fetching a batch of packets from the source.

   .. py:attribute:: single_packet_heaps

   Number of heaps that were entirely contained in a single packet. These
   take a slightly faster path as it is not necessary to reassemble them.

   .. py:attribute:: search_dist

   Number of hash table entries searched to find the heaps associated with
   packets. This is intended for debugging/profiling spead2 and **may be
   removed without notice**.

Additional statistics are available on the ringbuffer underlying the stream
(:attr:`~spead2.recv.Stream.ringbuffer` property), with similar caveats about
synchronisation.

.. py:class:: spead2.recv.Stream.Ringbuffer

   .. py:method:: size()

   Number of heaps currently in the ringbuffer.

   .. py:method:: capacity()

   Maximum number of heaps that can be held in the ringbuffer (corresponds to
   the `ring_heaps` argument to the stream constructor).
