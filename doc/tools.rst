Other tools
===========

.. _mcdump:

mcdump
------
mcdump is a tool similar to tcpdump_, but specialised for high-speed capture of
multicast UDP traffic using hardware that supports the Infiniband Verbs API. It
has only been tested on Mellanox ConnectX-3 NICs. Like gulp_, it uses a
separate thread for disk I/O and CPU core affinity to achieve reliable
performance.

It is not limited to capturing SPEAD data. It is included with spead2 rather
than released separately because it reuses a lot of the spead2 code.

.. _tcpdump: http://www.tcpdump.org/
.. _gulp: http://corey.elsewhere.org/gulp/

Installation
^^^^^^^^^^^^
The tool is automatically compiled and installed with spead2, provided that
libiverbs support is detected at configure time.

It may also be necessary to configure the system to work with ibverbs. See
:doc:`py-ibverbs` for more information.

Usage
^^^^^
The simplest incantation is

.. code-block:: sh

   mcdump -i xx.xx.xx.xx output.pcap yy.yy.yy.yy:zzzz

which will capture on the interface with IP address *xx.xx.xx.xx*, for the
multicast group *yy.yy.yy.yy* on UDP port *zzzz*. mcdump will take care of
subscribing to the multicast group. Note that only IPv4 is supported. Capture
continues until interrupted by :kbd:`Ctrl-C`. You can also list more
:samp:`{group}:{port}` pairs, which will all stored in the same pcap file.

Unfortunately, unlike tcpdump, it is not possible to tell directly tell whether
packets were dropped. NIC counters (on Linux, accessed with :command:`ethtool
-S`) can give an indication, although sometimes packets are dropped during the
shutdown process.

These options are important for performance:

.. option:: -N <cpu>, -C <cpu>, -D <cpu>

   Set CPU core IDs for various threads. The :option:`-D` option can be repeated
   multiple times to use multiple threads for disk I/O. By default, the threads
   are not bound to any particular core. It is recommended that these cores be
   on the same CPU socket as the NIC.

.. option:: --direct-io

   Use the ``O_DIRECT`` flag to open the file. This bypasses the kernel page
   cache, and can in some cases yield higher performance. However, not all
   filesystems support it, and it can also reduce performance when capturing
   a small enough amount of data that it will fit into RAM.

Limitations
^^^^^^^^^^^

- Packets are not timestamped (they all have a zero timestamp in the file).

- Only IPv4 is supported.
