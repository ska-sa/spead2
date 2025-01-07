Measuring sender performance
============================
Now that we have some functioning code, we're start using more features of
spead2 to improve performance. But first, we ought to have some idea of what
the performance is. We'll make some changes to observe the time before and
after we send the heaps, and also send more heaps (to make the timing more
reliable). We'll remove the target rate, so that we're just sending as fast as
we can. Setting the rate to 0 has the special meaning of removing any rate
limiting (it is also the default, so we could just not set it at all).

Unlike the previous sections though, we'll be modifying the code as we go,
rather than just writing it from top to bottom. If you're unclear on how
the shown snippets fit into the existing code, you can refer to the bottom of
the page for the full listing.

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

We'll also make a change that will make performance slightly more predictable:
pinning each thread to a specific CPU core. This avoids the costs incurred
(particularly related to L1 caches) when a process is migrated from one CPU
core to another.

.. tab-set-code::

 .. code-block:: python

    thread_pool = spead2.ThreadPool(1, [0])
    spead2.ThreadPool.set_affinity(1)

 .. code-block:: c++

    spead2::thread_pool thread_pool(1, {0});
    spead2::thread_pool::set_affinity(1);

The first line creates the thread pool with one thread, which is assigned to
core 0. The second line sets the affinity of the main thread (the function
lives in the thread pool namespace, but affects the current thread rather than
the thread pool). In other words, the thread pool has one thread bound to core
0 and the main Python thread is bound to core 1.

You can expect performance to be pretty low; I get around 65 MB/s from Python
and 140 MB/s from C++ [#benchmarks]_. In fact, spead2 makes very little
difference to the performance here: it's mostly taken up by generating the
random numbers. We don't actually care about the numbers being statistically
random, so let's remove the random number generation and replace it with the
following:

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

            item_group["adc_samples"].value = np.full(heap_size, np.int_(i), np.int8)

 .. code-block:: c++
    :dedent: 0

            std::fill(adc_samples.begin(), adc_samples.end(), i);

We're filling each heap with the loop index (truncated to an 8-bit integer) —
not particularly meaningful for simulation, but it has the bonus that we can
easily see on the receiver side if we've accidentally transmitted data for the
wrong heap.

This dramatically improves performance: around 1000 MB/s for Python and 1200
MB/s for C++ — assuming you're running the receiver. Somewhat surprisingly,
performance is much higher when not running the receiver: 1800 MB/s and 2200
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

.. tab-set-code::

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
            boost::asio::ip::make_address(argv[optind]),
            std::stoi(argv[optind + 1])
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
address assigned to the interface). I get 2300 MB/s with Python and 3000 MB/s
with C++.

.. [#benchmarks] I'll be quoting benchmark numbers throughout these tutorials.
   The numbers are what I encountered on my laptop at the time the tutorial was
   written, so they may be out of date with regards to future optimisations to
   spead2. They also vary each time I run them , and they will likely differ
   from what you encounter. I've also disabled Turbo Boost to reduce
   variability, but that significantly reduces the actual performance
   (top CPU speed drops from 4.5 GHz to 2.6 GHz).
   Treat them as rough indicators of how important various
   optimisations are, rather than as the absolute throughput you should expect
   from your application.

.. [#fakeaddr] This will not be the same as sending to an address of a real
   machine which is not listening on the chosen port: in that situation, the
   machine will send back ICMP "port unreachable" packets, which will affect
   the performance of the sending machine.

Full code
---------
.. tab-set-code::

   .. literalinclude:: ../../examples/tutorial/tut_4_send_perf.py
      :language: python

   .. literalinclude:: ../../examples/tutorial/tut_4_send_perf.cpp
      :language: c++
