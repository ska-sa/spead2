Support for ibverbs
===================
Receiver performance can be significantly improved by using the Infiniband
Verbs API instead of the BSD sockets API. This is currently only tested on
Linux with Mellanox ConnectX®-3 and ConnectX®-5 NICs. It depends on device
managed flow steering (DMFS), which may require using the Mellanox OFED version
of libibverbs.

There are a number of limitations in the current implementation:

 - Only IPv4 is supported
 - VLAN tagging, IP optional headers, and IP fragmentation are not supported
 - Only multicast is supported.

Within these limitations, it is quite easy to take advantage of this faster
code path. The main difficulty is that one *must* specify the IP address of
the interface that will send or receive the packets. The netifaces_ module can
help find the IP address for an interface by name.

.. _netifaces: https://pypi.python.org/pypi/netifaces

System configuration
--------------------
It is likely that some system configuration will be needed to allow this mode
to work correctly. For ConnectX®-3, add the following to
:file:`/etc/modprobe.d/mlnx.conf`::

   options ib_uverbs disable_raw_qp_enforcement=1
   options mlx4_core fast_drop=1
   options mlx4_core log_num_mgm_entry_size=-1

For more information, see the `libvma documentation`_. For ConnectX®-4/5, only
the first of these lines is required.

.. _libvma documentation: https://github.com/Mellanox/libvma

.. note::

   Setting ``log_num_mgm_entry_size`` to -7 instead of -1 will activate faster
   static device-managed flow steering. This has some limitations (refer to the
   manual_ for details), but can improve performance when capturing a large
   number of multicast groups.

   .. _manual: http://www.mellanox.com/related-docs/prod_software/Mellanox_EN_for_Linux_User_Manual_v4_3.pdf

Receiving
---------
The ibverbs API can be used programmatically by using an extra method of
:py:class:`spead2.recv.Stream`.

.. py:method:: spead2.recv.Stream.add_udp_ibv_reader(endpoints, interface_address, max_size=DEFAULT_UDP_IBV_MAX_SIZE, buffer_size=DEFAULT_UDP_IBV_BUFFER_SIZE, comp_vector=0, max_poll=DEFAULT_UDP_IBV_MAX_POLL)

      Feed data from multicast IPv4 traffic. For backwards compatibility, one
      can also pass a single address and port as two separate arguments in
      place of `endpoints`.

      :param list endpoints: List of 2-tuples, each containing a
        hostname/IP address the multicast group and the UDP port number.
      :param str interface_address: Hostname/IP address of the interface which
        will be subscribed
      :param int max_size: Maximum packet size that will be accepted
      :param int buffer_size: Requested memory allocation for work requests.
        It may be adjusted due to implementation-dependent limits or alignment
        requirements.
      :param int comp_vector: Completion channel vector (interrupt)
        for asynchronous operation, or
        a negative value to poll continuously. Polling
        should not be used if there are other users of the
        thread pool. If a non-negative value is provided, it
        is taken modulo the number of available completion
        vectors. This allows a number of readers to be
        assigned sequential completion vectors and have them
        load-balanced, without concern for the number
        available.
      :param int max_poll: Maximum number of times to poll in a row, without
        waiting for an interrupt (if `comp_vector` is
        non-negative) or letting other code run on the
        thread (if `comp_vector` is negative).

If supported by the NIC, the receive code will automatically use a
"multi-packet receive queue", which allows each packet to consume only the
amount of space needed in the buffer. This is not supported by all NICs (for
example, ConnectX-3 does not, but ConnectX-5 does). When in use, the
`max_size` parameter has little impact on performance, and is used only to
reject larger packets.

When multi-packet receive queues are not supported, performance can be
improved by making `max_size` as small as possible for the intended data
stream. This will increase the number of packets that can be buffered (because
the buffer is divided into fixed-size slots), and also improve memory
efficiency by keeping data more-or-less contiguous.

Environment variables
^^^^^^^^^^^^^^^^^^^^^
An existing application can be forced to use ibverbs for all multicast IPv4
readers, by setting the environment variable :envvar:`SPEAD2_IBV_INTERFACE` to the IP
address of the interface to receive the packets. Note that calls to
:py:meth:`spead2.recv.Stream.add_udp_reader` that pass an explicit interface
will use that interface, overriding :envvar:`SPEAD2_IBV_INTERFACE`; in this case,
:envvar:`SPEAD2_IBV_INTERFACE` serves only to enable the override.

It is also possible to specify :envvar:`SPEAD2_IBV_COMP_VECTOR` to override the
completion channel vector from the default.

Note that this environment variable currently has no effect on senders.

Sending
-------
Sending is done by using the class :py:class:`spead2.send.UdpIbvStream` instead
of :py:class:`spead2.send.UdpStream`. It has a different constructor, but the
same methods. There are also :py:class:`spead2.send.asyncio.UdpIbvStream` and
:py:class:`spead2.send.trollius.UdpIbvStream` classes, analogous to
:py:class:`spead2.send.asyncio.UdpStream` and
:py:class:`spead2.send.trollius.UdpStream`.

.. py:class:: spead2.send.UdpIbvStream(thread_pool, multicast_group, port, config, interface_address, buffer_size, ttl=1, comp_vector=0, max_poll=DEFAULT_MAX_POLL)

   Create a multicast IPv4 UDP stream using the ibverbs API

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param str multicast_group: Multicast group hostname/IP address
   :param int port: Destination port
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`
   :param str interface_address: Hostname/IP address of the interface which
     will be subscribed
   :param int buffer_size: Socket buffer size. A warning is logged if this
     size cannot be set due to OS limits.
   :param int ttl: Multicast TTL
   :param int buffer_size: Requested memory allocation for work requests.
   :param int comp_vector: Completion channel vector (interrupt)
     for asynchronous operation, or
     a negative value to poll continuously. Polling
     should not be used if there are other users of the
     thread pool. If a non-negative value is provided, it
     is taken modulo the number of available completion
     vectors. This allows a number of streams to be
     assigned sequential completion vectors and have them
     load-balanced, without concern for the number
     available.
   :param int max_poll: Maximum number of times to poll in a row, without
     waiting for an interrupt (if `comp_vector` is
     non-negative) or letting other code run on the
     thread (if `comp_vector` is negative).
