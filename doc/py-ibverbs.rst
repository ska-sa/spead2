Support for ibverbs
===================
Receiver performance can be significantly improved by using the Infiniband
Verbs API instead of the BSD sockets API. This is currently only tested on
Linux with ConnectX®-5 NICs. It depends on device managed flow steering
(DMFS).

There are a number of limitations in the current implementation:

 - Only IPv4 is supported.
 - VLAN tagging, IP optional headers, and IP fragmentation are not supported.
 - For sending, only multicast is supported.

Within these limitations, it is quite easy to take advantage of this faster
code path. The main difficulties are that one *must* specify the IP address of
the interface that will send or receive the packets, and that the
``CAP_NET_RAW`` capability may be needed. The netifaces_ module can
help find the IP address for an interface by name.

.. _netifaces: https://pypi.python.org/pypi/netifaces

System configuration
--------------------

ConnectX®-3
^^^^^^^^^^^
Add the following to :file:`/etc/modprobe.d/mlnx.conf`::

   options ib_uverbs disable_raw_qp_enforcement=1
   options mlx4_core fast_drop=1
   options mlx4_core log_num_mgm_entry_size=-1

.. note::

   Setting ``log_num_mgm_entry_size`` to -7 instead of -1 will activate faster
   static device-managed flow steering. This has some limitations (refer to the
   manual_ for details), but can improve performance when capturing a large
   number of multicast groups.

   .. _manual: http://www.mellanox.com/related-docs/prod_software/Mellanox_EN_for_Linux_User_Manual_v4_3.pdf

ConnectX®-4+, MLNX OFED up to 4.9
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Add the following to :file:`/etc/modprobe.d/mlnx.conf`::

   options ib_uverbs disable_raw_qp_enforcement=1

All other cases
^^^^^^^^^^^^^^^
No system configuration is needed, but the ``CAP_NET_RAW`` capability is
required. Running as root will achieve this; a full discussion of Linux
capabilities is beyond the scope of this manual.
For more information, see the `libvma documentation`_.

.. _libvma documentation: https://docs.mellanox.com/category/vma

Receiving
---------
The ibverbs API can be used programmatically by using an extra method of
:py:class:`spead2.recv.Stream`.

The configuration is specified using a :py:class:`spead.recv.UdpIbvConfig`.

.. py:class:: spead2.recv.UdpIbvConfig(*, endpoints=[], interface_address='', buffer_size=DEFAULT_BUFFER_SIZE, max_size=DEFAULT_MAX_SIZE, comp_vector=0, max_poll=DEFAULT_MAX_POLL)

   :param endpoints: Peer endpoints
   :type endpoints: List[Tuple[str, int]]
   :param str interface_address: Hostname/IP address of the interface which
     will be subscribed
   :param int buffer_size: Requested memory allocation for work requests.
     It may be adjusted to an integer number of packets.
   :param int max_size: Maximum packet size that will be accepted
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

   The constructor arguments are also instance attributes. Note that
   they are implemented as properties that return copies of the state, which
   means that mutating `endpoints` (for example, with :py:meth:`~list.append`)
   will not have any effect as only the copy will be modified. The entire list
   must be assigned to update it.

.. py:method:: spead2.recv.Stream.add_udp_ibv_reader(config)

   Feed data from IPv4 traffic.

If supported by the NIC and the drivers, the receive code will automatically
use a "multi-packet receive queue", which allows each packet to consume only
the amount of space needed in the buffer. This is currently only supported on
ConnectX®-4+ with MLNX OFED drivers 5.0 or later (or upstream rdma-core). When
in use, the `max_size` parameter has little impact on performance, and is used
only to reject larger packets.

When multi-packet receive queues are not supported, performance can be
improved by making `max_size` as small as possible for the intended data
stream. This will increase the number of packets that can be buffered (because
the buffer is divided into fixed-size slots), and also improve memory
efficiency by keeping data more-or-less contiguous.

Environment variables
^^^^^^^^^^^^^^^^^^^^^
An existing application can be forced to use ibverbs for all IPv4
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
same methods. There is also a :py:class:`spead2.send.asyncio.UdpIbvStream`
class, analogous to :py:class:`spead2.send.asyncio.UdpStream`.

There is an additional configuration class for ibverbs-specific
configuration:

.. py:class:: spead2.send.UdpIbvConfig(*, endpoints=[], interface_address='', buffer_size=DEFAULT_BUFFER_SIZE, ttl=1, comp_vector=0, max_poll=DEFAULT_MAX_POLL, memory_regions=[])

   :param endpoints: Peer endpoints (one per substream)
   :type endpoints: List[Tuple[str, int]]
   :param str interface_address: Hostname/IP address of the interface which
     will be subscribed
   :param int buffer_size: Requested memory allocation for work requests.
     It may be adjusted to an integer number of packets.
   :param int ttl: Multicast TTL
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
   :param List[object] memory_regions: Objects implementing the buffer
     protocol that will be used to hold item data. This is not required, but
     data stored in these buffers may be transmitted directly without
     requiring a copy, yielding higher performance. There may be
     platform-specific limitations on the size and number of these buffers.

   The constructor arguments are also instance attributes. Note that
   they are implemented as properties that return copies of the state, which
   means that mutating `endpoints` or `memory_regions` (for example, with
   :py:meth:`~list.append`) will not have any effect as only the copy will be
   modified. The entire list must be assigned to update it.

.. py:class:: spead2.send.UdpIbvStream(thread_pool, config, udp_ibv_config)

   Create a multicast IPv4 UDP stream using the ibverbs API

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`
   :param udp_ibv_config: Additional stream configuration
   :type udp_ibv_config: :py:class:`spead2.send.UdpIbvConfig`
