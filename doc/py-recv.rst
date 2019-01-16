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

Blocking receive
^^^^^^^^^^^^^^^^
To do blocking receive, create a :py:class:`spead2.recv.Stream`, and add
transports to it with :py:meth:`~spead2.recv.Stream.add_buffer_reader`,
:py:meth:`~spead2.recv.Stream.add_udp_reader`,
:py:meth:`~spead2.recv.Stream.add_tcp_reader` or
:py:meth:`~spead2.recv.Stream.add_udp_pcap_file_reader`. Then either iterate over
it, or repeatedly call :py:meth:`~spead2.recv.Stream.get`.

.. py:class:: spead2.recv.Stream(thread_pool, bug_compat=0, max_heaps=4, ring_heaps=4, contiguous_only=True, incomplete_keep_payload_ranges=False)

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param int bug_compat: Bug compatibility flags (see :ref:`py-flavour`)
   :param int max_heaps: The number of partial heaps that can be live at one
     time. This affects how intermingled heaps can be (due to out-of-order
     packet delivery) before heaps get dropped.
   :param int ring_heaps: The capacity of the ring buffer between the network
     threads and the consumer. Increasing this may reduce lock contention at
     the cost of more memory usage.
   :param bool contiguous_only: If set to ``False``, incomplete heaps will be
     included in the stream as instances of :py:class:`.IncompleteHeap`. By default
     they are discarded and a warning is printed.
   :param bool incomplete_keep_payload_ranges: If set to ``True``, it is
     possible to retrieve information about which parts of the payload arrived
     in incomplete heaps, using :py:meth:`.IncompleteHeap.payload_ranges`.
   :raises ValueError: if `max_heaps` is zero.

   .. py:method:: set_memory_allocator(allocator)

      Set or change the memory allocator for a stream. See
      :ref:`py-memory-allocators` for details.

      :param allocator: New memory allocator
      :type allocator: :py:class:`spead2.MemoryAllocator`

   .. py:method:: set_memcpy(id)

      Set the method used to copy data from the network to the heap. The
      default is :py:const:`MEMCPY_STD`. This can be changed to
      :py:const:`MEMCPY_NONTEMPORAL`, which writes to the destination with a
      non-temporal cache hint (if SSE2 is enabled at compile time). This can
      improve performance with large heaps if the data is not going to be used
      immediately, by reducing cache pollution. Be careful when benchmarking:
      receiving heaps will generally appear faster, but it can slow down
      subsequent processing of the heap because it will not be cached.

      :param id: Identifier for the copy function
      :type id: {:py:const:`MEMCPY_STD`, :py:const:`MEMCPY_NONTEMPORAL`}

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
      :param socket.socket socket: If specified, this socket is used rather
        than a new one. The socket must be open but unbound. The caller must
        not use this socket any further, although it is not necessary to keep
        it alive. This is mainly useful for fine-tuning socket options such
        as multicast subscriptions.

        .. deprecated:: 1.9
           Use the overload that doesn't take a `buffer_size` or `bind_hostname`.

   .. py:method:: add_udp_reader(multicast_group, port, max_size=DEFAULT_UDP_MAX_SIZE, buffer_size=DEFAULT_UDP_BUFFER_SIZE, interface_address)

      Feed data from a UDP port with multicast (IPv4 only).

      :param str multicast_group: Hostname/IP address of the multicast group to subscribe to
      :param int port: UDP port number
      :param int max_size: Largest packet size that will be accepted.
      :param int buffer_size: Kernel socket buffer size. If this is 0, the OS
        default is used. If a buffer this large cannot be allocated, a warning
        will be logged, but there will not be an error.
      :param str interface_address: Hostname/IP address of the interface which
        will be subscribed, or the empty string to let the OS decide.

   .. py:method:: add_udp_reader(multicast_group, port, max_size=DEFAULT_UDP_MAX_SIZE, buffer_size=DEFAULT_UDP_BUFFER_SIZE, interface_index)

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

   .. py:attribute:: stop_on_stop_item

      By default, a heap containing a stream control stop item will terminate
      the stream (and that heap is discarded). In some cases it is useful to
      keep the stream object alive and ready to receive a following stream.
      Setting this attribute to ``False`` will disable this special
      treatment. Such heaps can then be detected with
      :meth:`~spead2.recv.Heap.is_end_of_stream`.

   .. py:attribute:: allow_unsized_heaps

      By default, spead2 caters for heaps without a `HEAP_LEN` item, and will
      dynamically extend the memory allocation as data arrives. However, this
      can be expensive, and ideally senders should include this item. Setting
      this attribute to ``False`` will cause packets without this item to be
      rejected.

Asynchronous receive
^^^^^^^^^^^^^^^^^^^^
Asynchronous I/O is supported through Python 3's :py:mod:`asyncio` module, as
well as through trollius_ (a Python 2 backport). It can be combined with other
asynchronous I/O frameworks like twisted_ and Tornado_.

The documentation below is for the :py:mod:`asyncio` interface; replace all
instances of ``asyncio`` with ``trollius`` if you're using trollius.

.. py:class:: spead2.recv.asyncio.Stream(\*args, \*\*kwargs, loop=None)

   See :py:class:`spead2.recv.Stream` (the base class) for other constructor
   arguments.

   :param loop: Default asyncio event loop for async operations. If not
     specified, uses the default asyncio event loop. Do not call
     `get_nowait` from the base class.

   .. py:method:: get(loop=None)

      Coroutine that yields the next heap, or raises :py:exc:`spead2.Stopped`
      once the stream has been stopped and there is no more data. It is safe
      to have multiple in-flight calls, which will be satisfied in the order
      they were made.

      :param loop: asyncio event loop to use, overriding constructor.

.. _trollius: http://trollius.readthedocs.io/
.. _twisted: https://twistedmatrix.com/trac/
.. _tornado: http://www.tornadoweb.org/en/stable/

When using Python 3.5 or higher, the stream is also asynchronously iterable,
i.e., can be used in an ``async for`` loop to iterate over the heaps.

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

Incomplete Heaps
^^^^^^^^^^^^^^^^
By default, an incomplete heap (one for which some but not all of the packets
were received) are simply dropped and a warning is printed. Advanced users
might need finer control, such as recording metrics about the number of these
heaps. To do so, set `contiguous_only` to ``False`` when constructing the
stream. The stream will then yield instances of :py:class:`IncompleteHeap`.

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
      passed to the stream constructor; otherwise the information is dropped
      to save memory.

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
