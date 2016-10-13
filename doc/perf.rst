Performance tuning
==================
While spead2 tries to be performant out of the box, there are a number of ways
one can tune both the system and the application using spead2. It is usually
necessary to do at least some of these steps to achieve performance of
10Gb/s+, but your mileage may vary depending on your hardware and
application.

This guide focuses mostly on the problem of receiving data, because my
experience with high-bandwidth SPEAD has been with data produced by FPGAs.
Nevertheless, some of these tips also apply to sending data.

All advice is for a GNU/Linux system with an Intel CPU. You will need to
consult other documentation to find equivalent commands for other systems.

System tuning
-------------
The first thing to do is to increase the maximum socket buffer sizes. See
:doc:`introduction` for details.

The kernel firewall can affect performance, particularly if
small packets are not being used (in this context, anything that isn't a jumbo
frame is considered "small"). If possible, remove all firewall rules and
unload the kernel modules (those prefixed with ``ipt`` or ``nf``). In
particular, simply having the ``nf_conntrack`` module loaded can reduce
performance by several percent.

IP fragmentation also causes performance problems on the receiver. Check that
the routers in your network have a sufficiently large MTU that packets do not
get fragmented, particularly if using jumbo frames. You can use
:command:`tcpdump -v` to see fragments.

On a system with multiple CPU sockets, it is important to pin the process
using spead2 to a single socket, so that memory accesses do not cross the QPI
bus. For best performance, use the same socket as the NIC, which can be
determined from the output of :command:`hwloc-ls`. See :manpage:`numactl(8)`,
:manpage:`hwloc-ls(1)`, :manpage:`hwloc-bind(1)`.

There are a number of settings that can be adjusted to improve the system's
ability to respond to bursts of data. These will probably not improve peak
performance, but can reduce the number of lost heaps, particularly when a
stream starts and the system must ramp up performance in response.

- Disable hyperthreading.
- Disable CPU frequency scaling.
- Disable C states beyond C1 (for example, by passing
  ``intel_idle.max_state=1`` to the Linux kernel). Disabling
  C1 as well may reduce latency, but will likely limit the gains from Turbo
  Boost.
- Investigate disabling the P-state driver by passing ``intel_pstate=disable``
  on the kernel command line. The P-state driver has sometimes been reported
  to be much slower [#pstate1]_, [#pstate2]_, but can also be faster
  [#pstate3]_.
- Disable adaptive interrupt moderation on the NIC: :samp:`ethtool
  -C {interface} adaptive-rx off adaptive-tx off`. You may then need to
  experiment to tune the interrupt moderation settings — consult
  :manpage:`ethtool(8)` for details.
- Disable Ethernet flow control: :samp:`ethtool -A {interface}
  rx off tx off`.
- Use the isolcpus_ kernel option to completely isolate some CPU cores from
  other tasks, and pin the receiver to those cores (I have not actually tried
  this).
- Use :manpage:`chrt(1)` to run the receiver with real-time scheduling (I have
  not actually tried this).

.. _isolcpus: https://codywu2010.wordpress.com/2015/09/27/isolcpus-numactl-and-taskset/
.. [#pstate1] https://www.phoronix.com/scan.php?page=article&item=intel_pstate_linux315
.. [#pstate2] https://www.phoronix.com/scan.php?page=article&item=linux-47-schedutil
.. [#pstate3] https://www.phoronix.com/scan.php?page=news_item&px=Linux-4.4-CPUFreq-P-State-Gov

Protocol design
---------------
If you are designing a new SPEAD-based protocol, you have an opportunity to
make design choices that will make it easier for the sender and/or receiver to
reach the desired performance.

Heap size
^^^^^^^^^
The primary influence comes from heap size. There is some degree of overhead
for every heap (particularly for a Python receiver), and very small heaps will
cause this overhead to dominate. Heaps smaller than 16KiB are not recommended.
Very large heaps that do not fit into CPU caches will also reduce performance,
but not excessively. Memory usage also depends on the heap size. A number of
application tuning techniques described below also depend on knowing the heap
payload size a priori; thus, it is good practice to communicate this the
receiver in some way, whether by sending the descriptor early in the SPEAD
stream or by an out-of-band method.

Packet size
^^^^^^^^^^^
Packet size is not strictly part of the protocol, but also has a large impact
on performance. For 10Gb/s or faster streams, jumbo frames are highly
recommended, although with the kernel bypass techniques described below), this
is far less of an issue.

When using spead2 on the send side, the default packet size is 1472 bytes,
which is a safe value for IPv4 in a standard Ethernet setup [#]_.
The packet size is set in the :py:class:`~spead2.send.StreamConfig`. You
should pick a packet size, that, when added to the overhead for IP and UDP
headers, does not exceed the MTU of the link. For example, with IPv4 and an
MTU of 9200, use a packet size of 9172.

.. [#] The UDP and IP header together add 28 bytes, bringing the IP packet to
   the conventional MTU of 1500 bytes.

Alignment
^^^^^^^^^
Because items directly reference the received data (where possible), it is
possible that data will be misaligned. While numpy allows this, it could make
access to the data inefficient. The sender should ensure that data are
aligned. The spead2 sending API currently does not provide a way to enforce
this, but using items with round sizes will help.

Endianness
^^^^^^^^^^
When using numpy builtin types, data are converted to native endian when they
are received, to allow for more efficient operations on them. This can
reduce the maximum rate at which packets are received. Thus, using the native
endian on the wire (little-endian for x86) will give better performance.

Data format
^^^^^^^^^^^
Item descriptors can be specified using either a `format` or a `dtype` (numpy
data type). In many common cases, either can be used, and performance on a
Python receiver should be the same (a PySPEAD receiver, however, will be much
faster with `dtype`). The `dtype` is the only way to use Fortran order or
little-endian. The `format` approach is easier for a C++ receiver to parse
(since it does not need to decode a Python literal). It also allows for a
wider variety of types (such as bit vectors), but encoding or decoding these
types in Python takes a very slow path.

Application tuning
------------------
This section describes a number of ways the application can be modified to
improve performance. Most of these tuning options can be explored using a
provided benchmarking tool which measures the sustained performance on a
connection. This makes it possible to quickly identify the techniques that
will make the most difference before implementing them.

There are two versions of the benchmarking tool: one implemented in Python
(:program:`spead2_bench.py`) and one in C++ (:program:`spead2_bench`), which
are installed by the corresponding installers. The examples show the Python
version, but the C++ version functions very similarly.

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

There are also separate :program:`spead2_send` and :program:`spead2_recv` (and
Python equivalents) programs. The former generates a stream of meaningless
data, while the latter consumes an existing stream and reports the heaps and
items that it finds. Apart from being useful for debugging a stream,
:program:`spead2_recv` has a similar plethora of command-line options for
tuning that allow for exploration.

Kernel bypass APIs
^^^^^^^^^^^^^^^^^^
There are two low-level kernel bypass networking APIs supported:
:doc:`ibverbs <py-ibverbs>` and :doc:`netmap <cpp-netmap>`. These provide a
zero-copy path from the NIC into the spead2 library, without the kernel being
involved. This can make a huge performance difference, particularly for small
packet sizes.

Of these, ibverbs is the recommended one: it can be used without
being a root user, it is supported by both the Python and C++ APIs, can be
used for both sending and receiving, can be used by multiple processes or
streams simultaneously, and in simple cases requires only an environment
variable to be set. The netmap support is no longer developed or tested.

These APIs are not free: they will only work with some NICs, require special
kernel drivers and setup, have limitations in what networking features they
can support, and require the application to specify which network device to
use. Refer to the links above for more details.

Memory allocation
^^^^^^^^^^^^^^^^^
Using a :ref:`memory pool <py-memory-allocators>` is the single most important
tool for fast and reliable data transfer. It is particularly important when
heap sizes are large enough that :c:func:`malloc` and :c:func:`free` use
:c:func:`mmap` (:envvar:`M_MMAP_THRESHOLD` in glibc). For very small heaps,
memory pooling may be a net loss.

To use a memory pool, it is necessary to know the maximum heap payload size (a
conservative estimate is fine too — you will just use more memory). You also
need to size the pool appropriately. It is possible to specify a small
initial size and a larger maximum; however, each time the pool grows the CPU
will be busy with allocation and may drop packets. To avoid starvation, you
will need to provide:

- A buffer per partial heap (`max_heaps` parameter to
  :py:class:`spead2.recv.Stream`)
- A buffer per complete heap in the ring buffer (`ring_heaps` parameter to
  :py:class:`spead2.recv.Stream`)
- A buffer for every heap that has been taken off the ring buffer but not yet
  destroyed.
- A few extra for heaps that are in-flight between queues. The exact number
  may vary between releases, but 4 should be safe.

In general, it is best to err on the side of adding a few extra, provided that
this does not consume too much memory. At present there are unfortunately no
good tools for analysing memory pool performance.

Heap lifetime (Python)
~~~~~~~~~~~~~~~~~~~~~~
All the payload for a heap is stored in a single memory allocation, and where
possible, items reference this memory. This means that the entire heap remains
live as long as any of the values encoded in it are live. Thus, a small but
seldom-changing value can cause a very large heap to remain live long after
the rest of the values in that heap have been replaced. This can waste memory,
and also affects memory pool sizing.

To avoid this, senders should try to group items together that are updated at
the same frequency, rather than mixing low- and high-frequency items in the
same heap. Receivers can avoid this problem by copying values that are known to
be slowly varying.

Custom allocators (C++)
~~~~~~~~~~~~~~~~~~~~~~~
If you are doing an extra copy purely to put values into a special memory type
(for example, shared memory to communicate with another process, or pinned
memory for transfer to a GPU), then consider subclassing
:cpp:class:`spead2::memory_allocator`.

Tuning based on heap size
^^^^^^^^^^^^^^^^^^^^^^^^^
The library has a number of tuning parameters that are reasonable for
medium-to-large heaps (megabytes or larger). If using many
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

.. _perf-thread-pool:

Thread pools
^^^^^^^^^^^^
Each stream in spead2 has an associated thread pool, which provides worker
threads for handling incoming or outgoing packets. Each thread pool can have
some number of threads, defaulting to 1. Here are some rules of thumb:

- For a small number of streams (up to about the number of CPU cores), it is
  best to have one single-threaded thread pool per stream. This gives
  better cache affinity than a shared thread pool.
- For a large number of lower-bandwidth streams, use a shared thread pool with
  multiple threads. The number of threads should be chosen based on the number
  of CPU cores that you can dedicate to packet handling rather than other
  tasks in your application.
- A single stream cannot be processed by multiple threads at the same time, so
  there is never any benefit (and often detriment) to have more threads in a
  thread pool than there are streams serviced by that thread pool.
- Jitter (experienced as occasionally lost heaps) can be reduced by passing
  an affinity list to the thread pool constructor, to pin threads to specific
  cores. The main thread can be pinned as well, using
  :py:meth:`spead2.ThreadPool.set_affinity`.
