Sender, version 3
=================
In this section we'll make a small but significant optimisation, which as
usual can be found in :file:`examples/tut_send_3.py` and
:file:`examples/tut_send_3.cpp` in the spead2 repository. Currently we're
hampered by the speed of the random number generation, which is slower than
we're able to transmit. We don't particularly care about the values being
random, so let's delete all the random number code and replace it with the
following:

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

            item_group["adc_samples"].value = np.full(chunk_size, i, np.int8)

 .. code-block:: c++

            for (int j = 0; j < chunk_size; j++)
                adc_samples[j] = i;

We're filling each heap with the loop index — not particularly meaningful for
simulation, but it has the bonus that we can easily see on the receiver side
if we've accidentally transmitted data for the wrong heap. With this change,
performance improves to around 280 MB/s for Python and 310 MB/s for C++.

In :doc:`tut-spead-intro` we described heaps as the "messages" of SPEAD, but
we didn't dig much deeper into how these messages are transmitted on the wire.
Heaps can be extremely large, but lower-level protocols in the stack
(particularly Ethernet) place limits on how big frames or packets can be.
Standard Ethernet is limited to 1500-byte frames, while "jumbo" frames can be
up to 9 kiB. To accommodate these limitations, each SPEAD heap is split up
into one or more packets and reassembled on the receiver. There is overhead
associated with processing each packet, so for high performance, bigger
packets are better, provided you do not exceed the MTU (:dfn:`Maximum
Transmission Unit`) of the network path between sender and receiver.
Unfortunately, spead2 does not have any mechanism to discover the path MTU
[#mtu]_, so you will need to find it out and then pass the value into your
program.

Our example is still using the loopback interface, which has an MTU of 65536
bytes, but to produce something representative of Ethernet jumbo frames we'll
set a maximum packet size of 9000 bytes. Note that the packet size set in
spead2 contains only the SPEAD-specific parts of the packet, and does not
count the Ethernet, IP or UDP headers. For Ethernet with no VLAN information
and IPv4 with no options, those overheads come to 42 bytes, so this code is
assuming an MTU of at least 9042 bytes.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        config = spead2.send.StreamConfig(rate=0.0, max_heaps=2, max_packet_size=9000)

 .. code-block:: c++
    :dedent: 0

        config.set_max_packet_size(9000);

.. [#mtu] It would be effectively impossible in the case of multicast, since
   new subscribers could join at any time.
