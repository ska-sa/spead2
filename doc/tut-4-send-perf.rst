Measuring sender performance
============================
Now that we have some functioning code, we're start using more features of
spead2 to improve performance. But first, we ought to have some idea of what
the performance is. We'll make some changes to observe the time before and
after we send the heaps, and also send more heaps (to make the timing more
reliable). We'll remove the target rate, so that we're just sending as fast as
we can. Setting the rate to 0 has the special meaning of removing any rate
limiting (it is also the default, so we could just not set it at all).

The final code for this section can be found in
:file:`examples/tut_4_send_perf.py` and
:file:`examples/tut_4_send_perf.cpp`. Unlike the previous sections though,
we'll be modifying the code as we go, rather than just writing it from top to
bottom.

.. tab-set-code::

 .. code-block:: python

    import time
    ...
        config = spead2.send.StreamConfig(rate=0.0)
        ...
        n_heaps = 100
        start = time.perf_counter()
        for i in range(n_heaps):
            ...
        elapsed = time.perf_counter() - start
        print(f"{heap_size * n_heaps / elapsed / 1e6:.2f} MB/s")

 .. code-block:: c++

    #include <chrono>
    #include <iostream>
    #include <algorithm>  // Not needed yet, but we'll use it later
    ...
        config.set_rate(0.0);
        ...
        const int n_heaps = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_heaps; i++)
        {
            ...
        }
        std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << heap_size * n_heaps / elapsed.count() / 1e6 << " MB/s\n";

You can expect performance to be pretty low; I get around 90 MB/s from Python
and 220 MB/s from C++ [#benchmarks]_. In fact, spead2 makes very little
difference to the performance here: it's mostly taken up by generating the
random numbers. We don't actually care about the numbers being statistically
random, so let's remove the random number generation and replace it with the
following:

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

            item_group["adc_samples"].value = np.full(heap_size, i, np.int8)

 .. code-block:: c++
    :dedent: 0

            std::fill(adc_samples.begin(), adc_samples.end(), i);

We're filling each heap with the loop index — not particularly meaningful for
simulation, but it has the bonus that we can easily see on the receiver side
if we've accidentally transmitted data for the wrong heap.

This dramatically improves performance: around 1400 MB/s for Python and 1600
MB/s for C++ — assuming you're running the receiver. Somewhat surprisingly,
performance is much higher when not running the receiver: 2700 MB/s and 3200
MB/s respectively. By using the loopback interface, some of the costs of
receiving data are affecting the sender. Even when not running the receiver,
we're going to experience some overheads from using the loopback interface.

In a future tutorial we'll return to the receiver to improve its performance,
but for now let's move away from the loopback interface so that we can measure
the sender's performance in isolation. That means we'll need to transmit
packets somewhere other than to 127.0.0.1. Rather than hard-coding an address
(which may have a pre-assigned meaning on your network), let's make the
destination a command-line option. While we're at it, we'll make the number of
heaps a command-line option too, and increase the default.

 .. code-block:: python

    import argparse
    ...
    async def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("-n", "--heaps", type=int, default=1000)
        parser.add_argument("host", type=str)
        parser.add_argument("port", type=int)
        args = parser.parse_args()
        ...
        stream = spead2.send.UdpStream(thread_pool, [(args.host, args.port)], config)
        ...
        n_heaps = args.heaps

 .. code-block:: c++

    #include <unistd.h>
    ...
    static void usage(const char * name)
    {
        std::cerr << "Usage: " << name << " [-n heaps] host port\n";
    }

    int main(int argc, char * const argv[])
    {
        int opt;
        int n_heaps = 1000;  // remove the original definition of n_heaps
        while ((opt = getopt(argc, argv, "n:")) != -1)
        {
            switch (opt)
            {
            case 'n':
                n_heaps = std::stoi(optarg);
                break;
            default:
                usage(argv[0]);
                return 2;
            }
        }
        if (argc - optind != 2)
        {
            usage(argv[0]);
            return 2;
        }
        ...
        boost::asio::ip::udp::endpoint endpoint(
            boost::asio::ip::address::from_string(argv[optind]),
            std::atoi(argv[optind + 1])
        );
        ...
    }

The C++ version uses very quick-n-dirty command-line parsing; in a production
application you would need to do more error handling, and may want to use a
more modern library for it.

If you have a high-speed network interface, you can try sending to a
non-existent address on that network [#fakeaddr]_. But there is a portable
solution on Linux: a dummy interface. This is a network device that simply
drops all the data sent to it, instead of transmitting it anywhere. As such,
it represents an upper bound for what you're likely to achieve with kernel
drivers for real network interfaces.

You'll need a subnet to assign to it which isn't otherwise in use. For the
examples I'll use 192.168.31.0/24. You can configure a dummy interface like
this (as root):

.. code-block:: sh

   ip link add dummy1 type dummy
   ip addr add 192.168.31.1/24 dev dummy1
   ip link set dummy1 up

If you want to clean up the dummy interface later, use

.. code-block:: sh

   ip link del dummy1

Now if you run :command:`tut_4_send_perf 192.168.31.2 8888` you should get even
better performance (note that the destination address is *not* the same as the
address assigned to the interface). I get 3700 MB/s with Python and 4300 MB/s
with C++.
