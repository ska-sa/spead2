Performance tips
================
The :file:`tests` directory contains a script, :file:`spead2_bench.py`, that
can be used to measure the sustainable performance on a connection, and allows
many of the tunable parameters to be adjusted. On the receiver, pick a port
number (which must be free for both TCP and UDP) and run

.. code-block:: sh

   python spead2_bench.py slave <port>

Then, on the sender, run

.. code-block:: sh

   python spead2_bench.py master [options] <host> <port>

where *host* is the hostname of the receiver. This script will run tests at a
variety of speeds to determine the maximum speed at which the connection seems
reliable most of the time. This speed is right at the edge of stability: for a
totally reliable setup, you should use a slightly lower speed.

There is an equivalent :program:`spead2_bench` program using the C++ API in the
:file:`src` directory, which is built by the Makefile.

Packet size
-----------
The default packet size in a :py:class:`~spead2.send.StreamConfig` is 1472
bytes, which is a safe value for IPv4 in a standard Ethernet setup [#]_.
However, bigger packets significantly reduce overhead and are vital for
exceeding 10Gbps. You should pick a packet size, that, when added to the
overhead for IP and UDP headers, equals the MTU of the link. For example, with
IPv4 and an MTU of 9200, use a packet size of 9172.

.. [#] The UDP and IP header together add 28 bytes, bringing the IP packet to
   the conventional MTU of 1500 bytes.

Tuning based on heap size
-------------------------
The library has a number of tuning parameters that are reasonable for
medium-to-large heaps (megabytes to hundreds of megabytes). If using many
smaller heaps, some of the tuning parameters may need to be adjusted. In
particular

- Increase the `max_heaps` parameter to the
  :py:class:`spead2.send.StreamConfig` constructor.
- Increase the `max_heaps` parameter to the :py:class:`spead2.recv.Stream`
  constructor if you expect the network to reorder packets significantly
  (e.g., because data is arriving from multiple senders which are not
  completely synchronised). For single-packet heaps this has no effect.
- Increase the `ring_heaps` parameter to the :py:class:`spead2.recv.Stream`
  constructor to reduce lock contention. This has rapidly diminishing returns
  beyond about 16.

It is important to experiment to determine good values. Simply cranking
everything way up can actually reduce performance by increase memory usage and
thus reducing cache efficiency.

For very large heaps (gigabytes) some of these values can be decreased to 2
(or possibly even 1) to keep memory usage under control.

Heap lifetime
-------------
All the payload for a heap is stored in a single memory allocation, and where
possible, items reference this memory. This means that the entire heap remains
live as long as any of the values encoded in it are live. Thus, a small but
seldom-changing value can cause a very large heap to remain live long after
the rest of the values in that heap have been replaced. On the sending side
this can be avoided by only grouping items into a heap if they are updated at
the same frequency. If it is not possible to change the sender, the receiver
can copy values.

Alignment
---------
Because items directly reference the received data (where possible), it is
possible that data will be misaligned. While numpy allows this, it could make
access to the data inefficient. Ideally the sender will ensure that all values
are aligned within the heap payload, but unfortunately the bindings do not
currently provide a way to ensure this. If only a single addressed item is
placed in a heap, it will be naturally aligned.

Endianness
----------
When using numpy builtin types, data are converted to native endian when they
are received, to allow for more efficient operations on them. This can
significantly reduce the maximum rate at which packets are received. Thus,
using the native endian on the wire will give better performance.

Data format
-----------
Using the `dtype` parameter to the :py:class:`spead2.Item` constructor is
highly recommended. While the `format` parameter is more generic, it uses a
much slower path for encoding and decoding. In some cases it can determine an
equivalent `dtype` and use the fast path, but relying on this is not
recommended. The `dtype` approach is also the only way to transmit in
little-endian, which will be faster when the host is little-endian (such as
x86).

Hardware setup
--------------
Many systems are configured to drop to a lower-power state when idle, using
frequency scaling, C-states and so on. It can take hundreds of microseconds to
return to full performance, which can translate into megabytes of data on a
high-speed incoming stream. If a high-speed stream is expected, one should
disable these features or otherwise bring the system up to full performance
before the stream starts flowing.

Similarly, some Ethernet adaptors default to using adaptive interrupt
mitigation to dynamically adjust the latency/bandwidth tradeoff depending on
traffic, but can be too slow to adapt. Assuming your application is
latency-tolerant, this can be disabled with

.. code-block:: sh

    ethtool -C eth1 adaptive-rx off
