Sender, version 1
=================
We'll start by writing an example sender, which means we need to know
what we're sending. We'll simulate a digitiser, which means having a
continuous stream of real values (representing voltages). Since SPEAD is
message-based, we'll need to split the values into heaps. Additionally,
we'll want to timestamp the data, which we'll do just with a sample counter.

In this section we'll build up the code a piece at a time, for both
Python and C++. At the end of each section you will find the complete code for
reference. It can also be found in the :file:`examples/tutorial` directory in
the spead2 repository.

We'll start with some boilerplate:

.. tab-set-code::

 .. code-block:: python

    #!/usr/bin/env python3

    import numpy as np
    import spead2.send


    def main():

 .. code-block:: c++

    #include <cstdint>
    #include <random>
    #include <string>
    #include <vector>
    #include <utility>
    #include <boost/asio.hpp>
    #include <spead2/common_defines.h>
    #include <spead2/common_flavour.h>
    #include <spead2/common_thread_pool.h>
    #include <spead2/send_heap.h>
    #include <spead2/send_stream_config.h>
    #include <spead2/send_udp.h>

    int main()
    {

That just imports what we need. Next, we'll create a :dfn:`thread pool`. This
takes care of doing the actual networking with background threads. By default,
a thread pool has only 1 thread, and that's usually all you need.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        thread_pool = spead2.ThreadPool()

 .. code-block:: c++
    :dedent: 0

        spead2::thread_pool thread_pool;

Before creating the stream, we need to set up some configuration for it. For
performance reasons, spead2 doesn't let us change the configuration of a
stream after we've created it, so we first need to create a configuration
object. In Python we can set options either via keyword arguments to the
constructor (shown below) or via attribute access. The C++ version uses
getters and setters.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        config = spead2.send.StreamConfig(rate=100e6)

 .. code-block:: c++
    :dedent: 0

        spead2::send::stream_config config;
        config.set_rate(100e6);

Here we're setting the target transmission rate (in bytes per second),
although this code is not optimised so it won't necessarily achieve it. There
are other options that can be set, but we won't need them for this
example.

Now that we have the configuration, we can use it to create a stream. We'll
transmit the data over UDP, so we need to know where to send it. For this
tutorial we'll just hardcode an address (the local machine) and port number.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        stream = spead2.send.UdpStream(thread_pool, [("127.0.0.1", 8888)], config)

 .. code-block:: c++
    :dedent: 0

        boost::asio::ip::udp::endpoint endpoint(
            boost::asio::ip::make_address("127.0.0.1"),
            8888
        );
        spead2::send::udp_stream stream(thread_pool, {endpoint}, config);

Why is the destination not part of the config object? It is because that is
specific to the protocol used (UDP) while the configuration object is for
generic configuration (e.g., that is also applicable to in-process
communication). Astute readers might also notice that we pass a *list* of
endpoints. This is because spead2 allows different heaps within a stream to be
sent to different destinations.

We need to define the items that we will be transmitting. As mentioned
earlier, the Python API provides the ItemGroup class, which makes the code a
little simpler for this case.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        heap_size = 1024 * 1024
        item_group = spead2.send.ItemGroup()
        item_group.add_item(
            0x1600,
            "timestamp",
            "Index of the first sample",
            shape=(),
            format=[("u", spead2.Flavour().heap_address_bits)],
        )
        item_group.add_item(
            0x3300,
            "adc_samples",
            "ADC converter output",
            shape=(heap_size,),
            dtype=np.int8,
        )

 .. code-block:: c++
    :dedent: 0

        const std::int64_t heap_size = 1024 * 1024;
        spead2::descriptor timestamp_desc;
        timestamp_desc.id = 0x1600;
        timestamp_desc.name = "timestamp";
        timestamp_desc.description = "Index of the first sample";
        timestamp_desc.format.emplace_back('u', spead2::flavour().get_heap_address_bits());
        spead2::descriptor adc_samples_desc;
        adc_samples_desc.id = 0x3300;
        adc_samples_desc.name = "adc_samples";
        adc_samples_desc.description = "ADC converter output";
        adc_samples_desc.numpy_header =
            "{'shape': (" + std::to_string(heap_size) + ",), 'fortran_order': False, 'descr': 'i1'}";

There is quite a lot to take in here. We've arbitrarily assigned IDs 0x1600
for the timestamp and 0x3300 for the sample data. The SPEAD specification
recommends that user-defined IDs are at least 0x400. What is the upper limit?
Answering that requires understanding :dfn:`flavours` in SPEAD. When items are
encoded on the wire, the number of bytes used to encode the IDs is not fixed,
but rather specified in the packet header. The number of bits used to
represent certain fields such as the heap length (so-called :dfn:`immediate`
values) is also variable. The default flavour (which we will use here) is
called SPEAD-64-40, and allows for 23-bit item IDs and 40-bit immediate
values. The MeerKAT telescope largely uses SPEAD-64-48, which allows for
15-bit item IDs and 48-bit immediate values. In general, spead2 supports
SPEAD-64-N, where N is a multiple of 8, giving 63 - N bits for item
IDs and N bits for immediate values.

Let's look at types and shapes. For the ``timestamp`` we haven't set a shape,
so it defaults to scalar. The type is an unsigned integer (``u``
is defined in the SPEAD protocol to mean unsigned integer). The second part of
the ``format`` is the number of bits, which we're getting from
a default-constructed flavour object. This is the number of bits in an
immediate value — but what does that have to do with the timestamp?
A feature of the protocol is that values that have this number of bits can be
encoded in a more compact way. For this simple application it makes little
difference, but there are advanced use cases where it is important to use this
representation, which is why we illustrate it.

On the other hand, we've given ``adc_samples`` a one-dimensional shape, and
specified the type in a different way. We could have configured it similarly
to ``timestamp``, with format ``('i', 8)`` for 8-bit signed integer, but this
shows an alternative way to specify types in SPEAD, using the numpy type
system. In the C++ code, we have to manually construct the numpy format
header (it is described in :mod:`numpy.lib.format`) to include both the shape
and the type.

We're finally ready to start transmitting some data. For this tutorial we'll
just transmit synchronously, meaning that we'll completely transmit each heap
before preparing the next heap. We don't have any real analogue-to-digital
hardware to sample, so we'll just send random numbers between -100 and 100.
And we'll just send 10 heaps to keep things brief.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        rng = np.random.default_rng()
        for i in range(10):
            item_group["timestamp"].value = i * heap_size
            item_group["adc_samples"].value = rng.integers(-100, 100, size=heap_size, dtype=np.int8)
            heap = item_group.get_heap()
            stream.send_heap(heap)

 .. code-block:: c++
    :dedent: 0

        std::default_random_engine random_engine;
        std::uniform_int_distribution<std::int8_t> distribution(-100, 100);
        std::vector<std::int8_t> adc_samples(heap_size);

        for (int i = 0; i < 10; i++)
        {
            spead2::send::heap heap;
            // Add descriptors to the first heap
            if (i == 0)
            {
                heap.add_descriptor(timestamp_desc);
                heap.add_descriptor(adc_samples_desc);
            }
            // Create random data
            for (int j = 0; j < heap_size; j++)
                adc_samples[j] = distribution(random_engine);
            // Add the data and timestamp to the heap
            heap.add_item(timestamp_desc.id, i * heap_size);
            heap.add_item(
                adc_samples_desc.id,
                adc_samples.data(),
                adc_samples.size() * sizeof(adc_samples[0]),
                true
            );
            stream.async_send_heap(heap, boost::asio::use_future).get();
        }

The Python code is reasonably straight-forward: we update the items, package
the changes into a heap, and transmit it. The C++ code needs more explanation.
Firstly, as mentioned earlier, the Python API takes care of sending
descriptors in the first heap, so that the receiver knows the names, shapes
and types of the items. In C++ we must explicitly add the descriptors to the
first heap. The C++ code also uses two different versions of
:cpp:func:`~spead2::send::heap::add_item` to populate the data in the heap.
The first one takes the timestamp by value; it is only suitable for immediate
values. The second passes a pointer and a size and is more flexible.

We also said that we would be sending synchronously, but the C++ API only
provides an asynchronous send function. It uses the Boost `Asio`_ framework,
which means we can easily make it synchronous by passing the token
``boost::asio::use_future`` and then waiting for the returned future.

.. _Asio: https://www.boost.org/doc/libs/release/libs/asio/

Finally, we can consider what to do when we've ended the experiment and finished
sending data. We can send a special item in a heap to indicate that we're
finished and that the receiver can shut down. Since this is being sent over
UDP it is not 100% reliable and a real application should have a fallback
mechanism, but we'll ignore that for now. Note that the protocol also defines
a similar control item to indicate the start of the stream, but it is not as
useful (since the arrival of data implicitly indicates that it has started).

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        stream.send_heap(item_group.get_end())


    if __name__ == "__main__":
        main()

 .. code-block:: c++
    :dedent: 0

        spead2::send::heap heap;
        heap.add_end();
        stream.async_send_heap(heap, boost::asio::use_future).get();
        return 0;
    }

That's it! Let's give it a test. If you've been following the C++ tutorial,
you'll want a compiled binary. Assuming you've installed spead2, you should be
able to compile the example code with

.. code-block:: sh

   g++ -o tut_2_send tut_2_send.cpp -Wall -O3 `pkg-config --cflags --libs spead2`

If you installed spead2 into a non-standard location, you may need to set
:envvar:`PKG_CONFIG_PATH` to the directory containing the installed
:file:`spead2.pc`. Building spead2 from source also compiles the examples
in the :file:`examples/tutorial` subdirectory of the build directory.

Unfortunately, in the best case, running the code gives no output at all and
the program simply exits. Obviously, we're going to need a receiver to get
some idea of whether anything is really happening. The good news is that
spead2 ships with a general-purpose receiver — in fact two (one written in
Python and one written in C++). Let's use the Python one, since it provides
more high-level interpretation of the data. Note that you can use the Python
receiver even with the C++ sender, since the protocol is the same, although
if you haven't already :doc:`installed <../installation>` the Python bindings you
should do that now.

Start the receiver first by running

.. code-block:: sh

    spead2_recv.py --descriptors --values 127.0.0.1:8888

This will listen on port 8888 on the local machine — the same port our program
is sending to. Then run the example program again. The receiver program should
now print something like the following and exit:

.. code-block:: text

    Received heap 1 on stream 127.0.0.1:8888
        Descriptor for timestamp (0x1600)
          description: Index of the first sample
          format:      [('u', 40)]
          dtype:       None
          shape:       ()
        Descriptor for adc_samples (0x3300)
          description: ADC converter output
          format:      None
          dtype:       int8
          shape:       (1048576,)
    adc_samples = [ 63  55  23 ... -61  50 -82]
    timestamp = 0
    Received heap 2 on stream 127.0.0.1:8888
    adc_samples = [-28  33 -42 ... -25 -12  15]
    timestamp = 1048576
    Received heap 3 on stream 127.0.0.1:8888
    adc_samples = [-43 -14 -18 ... -12 -70 -61]
    timestamp = 2097152
    Received heap 4 on stream 127.0.0.1:8888
    adc_samples = [  79    2 -100 ...   59    6  -71]
    timestamp = 3145728
    Received heap 5 on stream 127.0.0.1:8888
    adc_samples = [ 38  -5  84 ... -67 -93  57]
    timestamp = 4194304
    Received heap 6 on stream 127.0.0.1:8888
    adc_samples = [ -4   1 -33 ... -99  96  15]
    timestamp = 5242880
    Received heap 7 on stream 127.0.0.1:8888
    adc_samples = [  5 -48 -46 ...  86  65 -59]
    timestamp = 6291456
    Received heap 8 on stream 127.0.0.1:8888
    adc_samples = [ 79 -38 -41 ... -22 -73   0]
    timestamp = 7340032
    Received heap 9 on stream 127.0.0.1:8888
    adc_samples = [  4 -40  84 ... -19 -11 -43]
    timestamp = 8388608
    Received heap 10 on stream 127.0.0.1:8888
    adc_samples = [  2 -64 -87 ...   0  84 -76]
    timestamp = 9437184
    Shutting down stream 127.0.0.1:8888 after 10 heaps
    heaps: 10
    incomplete_heaps_evicted: 0
    incomplete_heaps_flushed: 0
    packets: 7331
    batches: 2359
    max_batch: 45
    single_packet_heaps: 1
    search_dist: 7330
    worker_blocked: 0

We can see that the first heap contains the descriptors we set. All the
heaps contain a timestamp and some sample data (not fully shown). At the end
we see some :doc:`statistics <../recv-stats>`. Don't worry if you don't
understand them all; some of them are only intended to help developers or
advanced users diagnose performance bottlenecks.

Full code
---------
.. tab-set-code::

   .. literalinclude:: ../../examples/tutorial/tut_2_send.py
      :language: python

   .. literalinclude:: ../../examples/tutorial/tut_2_send.cpp
      :language: c++
