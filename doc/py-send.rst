Sending
=======
Unlike for receiving, each stream object can only use a single transport.
There is currently no support for collective operations where multiple
producers cooperate to construct a heap between them. It is still possible to
do multi-producer, single-consumer operation if the heap IDs are kept separate.

Because each stream has only one transport, there is a separate class for
each, rather than a generic `Stream` class. Because there is common
configuration between the stream classes, configuration is encapsulated in a
:py:class:`spead2.send.StreamConfig`.

.. py:class:: spead2.send.StreamConfig(max_packet_size=1472, rate=0.0, burst_size=65536, max_heaps=4, burst_rate_ratio=1.05)

   :param int max_packet_size: Heaps will be split into packets of at most this size.
   :param double rate: Target transmission rate, in bytes per second, or 0
     to send as fast as possible.
   :param int burst_size: Bursts of up to this size will be sent as fast as
     possible. Setting this too large (larger than available buffer sizes)
     risks losing packets, while setting it too small may reduce throughput by
     causing more sleeps than necessary.
   :param int max_heaps: For asynchronous transmits, the maximum number of
     heaps that can be in-flight.
   :param float burst_rate_ratio: If packet sending falls below the target
     transmission rate, the rate will be increased until the average rate
     has caught up. This value specifies the "catch-up" rate, as a ratio to the
     target rate.

   The constructor arguments are also instance attributes.

Streams send pre-baked heaps, which can be constructed by hand, but are more
normally created from an :py:class:`~spead2.ItemGroup` by a
:py:class:`spead2.send.HeapGenerator`. To simplify cases where one item group
is paired with one heap generator, a convenience class
:py:class:`spead2.send.ItemGroup` is provided that inherits from both.

.. autoclass:: spead2.send.HeapGenerator

   .. automethod:: spead2.send.HeapGenerator.add_to_heap
   .. automethod:: spead2.send.HeapGenerator.get_heap
   .. automethod:: spead2.send.HeapGenerator.get_start
   .. automethod:: spead2.send.HeapGenerator.get_end

.. py:class:: spead2.send.Heap(flavour=spead2.Flavour())

   .. py:attribute:: repeat_pointers

      Enable/disable repetition of item pointers in all packets.

      Usually this is not needed, but it can enable some specialised use
      cases where immediates can be recovered from incomplete heaps or where
      the receiver examines the item pointers in each packet to decide how
      to handle it. The packet size must be large enough to fit all the item
      pointers for the heap (the implementation also reserves a little space,
      so do not rely on a tight fit working).

      The default is disabled.

   .. py:method:: add_item(item)

      Add an :py:class:`~spead2.Item` to the heap. This references the memory in
      the item rather than copying it. It does *not* cause a descriptor to be
      sent; use :py:meth:`add_descriptor` for that.

   .. py:method:: add_descriptor(descriptor)

      Add a :py:class:`~spead2.Descriptor` to the heap.

   .. py:method:: add_start()

      Convenience method to add a start-of-stream item.

   .. py:method:: add_end()

      Convenience method to add an end-of-stream item.

Blocking send
-------------

There are multiple stream classes, corresponding to different transports, and
some of the classes have several variants of the constructor. They all
implement the following interface, although this base class does not actually exist:

.. py:class:: spead2.send.AbstractStream()

   .. py:method:: send_heap(heap, cnt=-1)

      Sends a :py:class:`spead2.send.Heap` to the peer, and wait for
      completion. There is currently no indication of whether it successfully
      arrived, but :py:exc:`IOError` is raised if it could not be sent.

      If not specified, a heap cnt is chosen automatically (the choice can be
      modified by calling :py:meth:`set_cnt_sequence`). If a non-negative value
      is specified for `cnt`, it is used instead. It is the user's
      responsibility to avoid collisions.

   .. py:method:: set_cnt_sequence(next, step)

      Modify the linear sequence used to generate heap cnts. The next heap
      will have cnt `next`, and each following cnt will be incremented by
      `step`. When using this, it is the user's responsibility to ensure
      that the generated values remain unique. The initial state is `next` =
      1, `step` = 1.

      This is useful when multiple senders will send heaps to the same
      receiver, and need to keep their heap cnts separate.

      If the computed cnt overflows the number of bits available, the
      bottom-most bits are taken.

UDP
^^^

Note that since UDP is an unreliable protocol, there is no guarantee that packets arrive.

.. py:class:: spead2.send.UdpStream(thread_pool, hostname, port, config=spead2.send.StreamConfig(), buffer_size=DEFAULT_BUFFER_SIZE, interface_address='')

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param str hostname: Peer hostname
   :param int port: Peer port
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`
   :param int buffer_size: Socket buffer size. A warning is logged if this
     size cannot be set due to OS limits.
   :param str interface_address: Source hostname/IP address (see tips about
     :ref:`routing`).

.. py:class:: spead2.send.UdpStream(thread_pool, multicast_group, port, config=spead2.send.StreamConfig(), buffer_size=DEFAULT_BUFFER_SIZE, ttl)

   Stream using UDP, with multicast TTL. Note that the regular constructor will
   also work with multicast, but does not give any control over the TTL.

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param str multicast_group: Multicast group hostname/IP address
   :param int port: Destination port
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`
   :param int buffer_size: Socket buffer size. A warning is logged if this
     size cannot be set due to OS limits.
   :param int ttl: Multicast TTL

.. py:class:: spead2.send.UdpStream(thread_pool, multicast_group, port, config=spead2.send.StreamConfig(), buffer_size=524288, ttl, interface_address)

   Stream using UDP, with multicast TTL and interface address (IPv4 only).

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param str multicast_group: Multicast group hostname/IP address
   :param int port: Destination port
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`
   :param int buffer_size: Socket buffer size. A warning is logged if this
     size cannot be set due to OS limits.
   :param int ttl: Multicast TTL
   :param str interface_address: Hostname/IP address of the interface on which
     to send the data

.. py:class:: spead2.send.UdpStream(thread_pool, multicast_group, port, config=spead2.send.StreamConfig(), buffer_size=DEFAULT_BUFFER_SIZE, ttl, interface_index)

   Stream using UDP, with multicast TTL and interface index (IPv6 only).

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param str multicast_group: Multicast group hostname/IP address
   :param int port: Destination port
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`
   :param int buffer_size: Socket buffer size. A warning is logged if this
     size cannot be set due to OS limits.
   :param int ttl: Multicast TTL
   :param str interface_index: Index of the interface on which to send the
     data

.. py:class:: spead2.send.UdpStream(thread_pool, socket, hostname, port, config=spead2.send.StreamConfig())

   Stream using UDP, with a pre-existing socket. The socket is duplicated by
   the stream, so the original can be closed immediately to free up a file
   descriptor. The caller is responsible for setting any socket options. The
   socket must not be connected.

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param socket.socket socket: UDP socket
   :param str hostname: Peer hostname
   :param int port: Peer port
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`

.. py:class:: spead2.send.UdpStream(thread_pool, hostname, port, config=spead2.send.StreamConfig(), buffer_size=DEFAULT_BUFFER_SIZE, socket)

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param str hostname: Peer hostname
   :param int port: Peer port
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`
   :param int buffer_size: Socket buffer size. A warning is logged if this
     size cannot be set due to OS limits.
   :param socket.socket socket: This socket is used rather
     than a new one. The socket must not be connected. The caller must
     not use this socket any further, although it is not necessary to keep
     it alive. This is mainly useful for fine-tuning socket options.

   .. deprecated:: 1.9
      Use the overload that does not take `buffer_size`.

TCP
^^^

TCP/IP is a reliable protocol, so heap delivery is guaranteed. However, if
multiple threads all call :py:meth:`~spead2.send.AbstractStream.send_heap` at
the same time, they can exceed the configured `max_heaps` and heaps will be dropped.

Because spead2 was originally designed for UDP, the default packet size in
:py:class:`~.StreamConfig` is quite small. Performance can be improved by
increasing it (but be sure the receiver is configured to handle larger packets).

.. py:class:: spead2.send.TcpStream(thread_pool, hostname, port, config=spead2.send.StreamConfig(), buffer_size=DEFAULT_BUFFER_SIZE, interface_address='')

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param str hostname: Peer hostname
   :param int port: Peer port
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`
   :param int buffer_size: Socket buffer size. A warning is logged if this
     size cannot be set due to OS limits.
   :param str interface_address: Source hostname/IP address (see tips about
     :ref:`routing`).

.. py:class:: spead2.send.TcpStream(thread_pool, socket, config=spead2.send.StreamConfig())

   Stream using an existing socket. The socket must already be connected to the
   peer, and the user is responsible for setting any desired socket options. The socket
   is duplicated, so it can be closed to save resources.

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param socket.socket socket: TCP socket
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`

Raw bytes
^^^^^^^^^

.. py:class:: spead2.send.BytesStream(thread_pool, config=spead2.send.StreamConfig())

   Stream that collects packets in memory and makes the concatenated stream
   available.

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`

   .. py:method:: getvalue()

      Return a copy of the memory buffer.

      :rtype: :py:class:`bytes`

In-process transport
^^^^^^^^^^^^^^^^^^^^
Refer to the separate :doc:`documentation <py-inproc>`.

.. _asynchronous-send:

Asynchronous send
-----------------

As for asynchronous receives, asynchronous sends are managed by asyncio_ or
trollius_. A
stream can buffer up multiple heaps for asynchronous send, up to the limit
specified by `max_heaps` in the :py:class:`~spead2.send.StreamConfig`. If this
limit is exceeded, heaps will be dropped, and the returned future has an
:py:exc:`IOError` exception set. An :py:exc:`IOError` could also indicate a
low-level error in sending the heap (for example, if the packet size exceeds
the MTU).

.. _trollius: http://trollius.readthedocs.io/
.. _asyncio: https://docs.python.org/3/library/asyncio.html

The classes existing in the :py:mod:`spead2.send.asyncio` and
:py:mod:`spead2.send.trollius` modules, and mostly implement the same
constructors as the synchronous classes. They implement the following abstract
interface (the class does not actually exist):

.. class:: spead2.send.asyncio.AbstractStream()

   .. py:method:: async_send_heap(heap, cnt=-1, loop=None)

      Send a heap asynchronously. Note that this is *not* a coroutine:
      it returns a future. Adding the heap to the queue is done
      synchronously, to ensure proper ordering.

      :param heap: Heap to send
      :type heap: :py:class:`spead2.send.Heap`
      :param int cnt: Heap cnt to send (defaults to auto-incrementing)
      :param loop: Event loop to use, overriding the constructor
      :type loop: :py:class:`asyncio.AbstractEventLoop`

   .. py:method:: flush

      Block until all enqueued heaps have been sent (or dropped).

   .. py:method:: async_flush

      Asynchronously wait for all enqueued heaps to be sent. Note that
      this only waits for heaps passed to :meth:`async_send_heap` prior to
      this call, not ones added while waiting.

TCP
^^^

For TCP, construction is slightly different: except when providing a custom
socket, one uses a coroutine to connect:

.. automethod:: spead2.send.asyncio.TcpStream.connect
