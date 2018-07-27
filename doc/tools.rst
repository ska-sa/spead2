Command-line tools
==================

.. _spead2_bench:

spead2_bench
------------
A benchmarking tool is provided to estimate the maximum throughput for UDP.
There are two versions: one implemented in Python (:program:`spead2_bench.py`)
and one in C++ (:program:`spead2_bench`), which are installed by the
corresponding installers. The examples show the Python version, but the C++
version functions very similarly. However, they cannot be mixed: use the same
version on each end of the connection.

On the receiver, pick a port number (which must be free for both TCP and UDP)
and run

.. code-block:: sh

   spead2_bench.py slave <port>

Then, on the sender, run

.. code-block:: sh

   spead2_bench.py master [options] <host> <port>

where *host* is the hostname of the receiver. This script will run tests at a
variety of speeds to determine the maximum speed at which the connection seems
reliable most of the time. This speed is right at the edge of stability: for a
totally reliable setup, you should use a lower speed.

spead2_send/spead2_recv
-----------------------
There are also separate :program:`spead2_send` and :program:`spead2_recv` (and
Python equivalents) programs. The former generates a stream of meaningless
data, while the latter consumes an existing stream and reports the heaps and
items that it finds. Apart from being useful for debugging a stream,
:program:`spead2_recv` has a similar plethora of command-line options for
tuning that allow for exploration.

.. _mcdump:

mcdump
------
mcdump is a tool similar to tcpdump_, but specialised for high-speed capture of
multicast UDP traffic using hardware that supports the Infiniband Verbs API. It
has only been tested on Mellanox ConnectX-3 NICs. Like gulp_, it uses a
separate thread for disk I/O and CPU core affinity to achieve reliable
performance. With a sufficiently fast disk subsystem, it is able to capture
line rate from a 40Gb/s adapter.

It is not limited to capturing SPEAD data. It is included with spead2 rather
than released separately because it reuses a lot of the spead2 code.

.. _tcpdump: http://www.tcpdump.org/
.. _gulp: http://corey.elsewhere.org/gulp/

Installation
^^^^^^^^^^^^
The tool is automatically compiled and installed with spead2, provided that
libibverbs support is detected at configure time.

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

You can also specify ``-`` in place of the filename to suppress the write to
file. This is useful to simply count the bytes/packets received without being
limited by disk throughput.

Unfortunately, unlike tcpdump, it is not possible to directly tell whether
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

.. option:: --count <count>

   Stop after <count> packets have been received. Without this option, mcdump
   will run until SIGINT (Ctrl-C) is received.

Limitations
^^^^^^^^^^^

- Timestamps are only collected if Mellanox extensions to the verbs API are
  detected at compile time. Otherwise, all packets have a zero timestamp in the
  file.

- Only IPv4 multicast is supported.

- It is not optimised for small packets (below about 1KB). Packet capture rates
  top out around 6Mpps for current hardware.
