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

            adc_samples.resize(chunk_size, i);

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

When we benchmark this, the performance has dramatically improved (over 1
GB/s), but we also find something surprising: performance is substantially
higher if we are listening for the data than if we are not. This is almost
certainly because sending a UDP packet to a port that is not open for
receiving causes an ICMP error packet to be generated, which reduces
performance. Even when we do attach a receiver, we're not seeing the full
performance, because the Linux loopback interface generally blocks the sender
if the receiver is not keeping up, rather than just dropping the packets.

In a future tutorial we'll return to the receiver to improve its performance,
but for now let's move away from the loopback interface so that we can measure
the sender's performance in isolation. That means we'll need to transmit
packets somewhere other than to 127.0.0.1. Rather than hard-coding an address
(which may have a pre-assigned meaning on your network), let's make the
destination a command-line option. While we're at it, let's increase the
number of heaps to get more reliable measurements.

.. tab-set-code::

 .. code-block:: python

    import argparse
    ...
    async def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("host", type=str)
        parser.add_argument("port", type=int)
        args = parser.parse_args()
        ...
        stream = spead2.send.asyncio.UdpStream(thread_pool, [(args.host, args.port)], config)
        ...
        n_heaps = 10000

 .. code-block:: c++

    int main(int argc, char * const argv[])
    {
        if (argc != 3)
        {
            std::cerr << "Usage: " << argv[0] << " <address> <port>\n";
            return 2;
        }
        ...
        boost::asio::ip::udp::endpoint endpoint(
            boost::asio::ip::address::from_string(argv[1]),
            std::atoi(argv[2])
        );
        ...
        const int n_heaps = 10000;

The C++ version uses very quick-n-dirty parsing of the IP address and port;
in a production application you would need to do more error handling.

If you have a high-speed network interface, you can try sending to a
non-existent address on that network. But there is a portable solution on
Linux: a dummy interface. You'll need a subnet to assign to it which isn't
otherwise in use. For the examples I'll use 192.168.31.0/24. You can
configure a dummy interface like this (as root):

.. code-block:: sh

   ip link add dummy1 type dummy
   ip link set mtu 9216 dev dummy1
   ip addr add 192.168.31.1/24 dev dummy1
   ip link set dummy1 up

Now if you run :command:`tut_send_3 192.168.31.2 8888` you should get even
better performance. I get around 3500 MB/s (with either C++ or Python), which
is getting close to the limit of what can be achieved for a single thread with
the kernel networking stack. Exceeding this will require either using multiple
multiple spead2 stream objects (each with their own thread pool), or
specialised network hardware.

Note that the destination address (192.168.31.2) is *not* the same as the
address we assigned to the interface; we want to send to an address that
doesn't exist, so that the packets are simply dropped.

If you want to clean up the dummy interface afterwards, use

.. code-block:: sh

   ip link del dummy1
