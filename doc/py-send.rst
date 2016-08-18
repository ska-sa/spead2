Sending
-------
Unlike for receiving, each stream object can only use a single transport.
There is currently no support for collective operations where multiple
producers cooperate to construct a heap between them. It is still possible to
do multi-producer, single-consumer operation if the heap IDs are kept separate.

Because each stream has only one transport, there is a separate class for
each, rather than a generic `Stream` class. Because there is common
configuration between the stream classes, configuration is encapsulated in a
:py:class:`spead2.send.StreamConfig`.

.. py:class:: spead2.send.StreamConfig(max_packet_size=1472, rate=0.0, burst_size=65536, max_heaps=4)

   :param int max_packet_size: Heaps will be split into packets of at most this size.
   :param double rate: Maximum transmission rate, in bytes per second, or 0
     to send as fast as possible.
   :param int burst_size: Bursts of up to this size will be sent as fast as
     possible. Setting this too large (larger than available buffer sizes)
     risks losing packets, while setting it too small may reduce throughput by
     causing more sleeps than necessary.
   :param int max_heaps: For asynchronous transmits, the maximum number of
     heaps that can be in-flight.

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

Blocking send
^^^^^^^^^^^^^

.. py:class:: spead2.send.UdpStream(thread_pool, hostname, port, config, buffer_size=DEFAULT_BUFFER_SIZE, socket=None)

   Stream using UDP. Note that since UDP is an unreliable protocol, there is
   no guarantee that packets arrive.

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param str hostname: Peer hostname
   :param int port: Peer port
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`
   :param int buffer_size: Socket buffer size. A warning is logged if this
     size cannot be set due to OS limits.
   :param socket.socket socket: If specified, this socket is used rather
     than a new one. The socket must be open but unbound. The caller must
     not use this socket any further, although it is not necessary to keep
     it alive. This is mainly useful for fine-tuning socket options.

   .. py:method:: send_heap(heap, cnt=-1)

      Sends a :py:class:`spead2.send.Heap` to the peer, and wait for
      completion. There is currently no indication of whether it successfully
      arrived.

      If not specified, a heap cnt is chosen automatically (the choice can be
      modified by calling :py:meth:`set_cnt_sequence`). If a non-negative value
      is specified for `cnt`, it is used instead. It is the user's
      responsibility to avoid collisions.

    .. py:method:: set_cnt_sequence(next, step)

       Modify the linear sequence used to generate heap cnts. The next heap
       will have cnt `next`, and each following cnt will be incremented by
       `step`. When using this, it is the user's responsibility to ensure
       that the generated values remain unique. The initial state is `next` =
       1, `cnt` = 1.

       This is useful when multiple senders will send heaps to the same
       receiver, and need to keep their heap cnts separate.

.. py:class:: spead2.send.UdpStream(thread_pool, multicast_group, port, config, buffer_size=DEFAULT_BUFFER_SIZE, ttl)

   Stream using UDP, with multicast TTL. Note that the regular constructor will
   also work with UDP, but does not give any control over the TTL.

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param str multicast_group: Multicast group hostname/IP address
   :param int port: Destination port
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`
   :param int buffer_size: Socket buffer size. A warning is logged if this
     size cannot be set due to OS limits.
   :param int ttl: Multicast TTL

.. py:class:: spead2.send.UdpStream(thread_pool, multicast_group, port, config, buffer_size=524288, ttl, interface_address)

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

.. py:class:: spead2.send.UdpStream(thread_pool, multicast_group, port, config, buffer_size=524288, ttl, interface_index)

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

.. py:class:: spead2.send.BytesStream(thread_pool, config)

   Stream that collects packets in memory and makes the concatenated stream
   available.

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`

   .. py:method:: send_heap(heap)

      Appends a :py:class:`spead2.send.Heap` to the memory buffer.

   .. py:method:: getvalue()

      Return a copy of the memory buffer.

      :rtype: :py:class:`bytes`


Asynchronous send
^^^^^^^^^^^^^^^^^

As for asynchronous receives, asynchronous sends are managed by trollius_. A
stream can buffer up multiple heaps for asynchronous send, up to the limit
specified by `max_heaps` in the :py:class:`~spead2.send.StreamConfig`. If this
limit is exceeded, heaps will be dropped, and the returned future has an
:py:exc:`IOError` exception set. An :py:exc:`IOError` could also indicate a
low-level error in sending the heap (for example, if the packet size exceeds
the MTU).

.. _trollius: http://trollius.readthedocs.io/

.. autoclass:: spead2.send.trollius.UdpStream(thread_pool, hostname, port, config, buffer_size=524288, socket=None, loop=None)

   .. automethod:: spead2.send.trollius.UdpStream.async_send_heap
   .. py:method:: flush

      Block until all enqueued heaps have been sent (or dropped).

   .. automethod:: spead2.send.trollius.UdpStream.async_flush
