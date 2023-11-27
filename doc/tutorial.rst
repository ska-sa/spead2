Tutorial
========
This chapter will walk you through creating basic sender and receiver
applications. The resulting application will *not* be high-performance, and
we will not cover all the features of the library.

A quick introduction to SPEAD
-----------------------------
Before developing a full-fledged application it is a good idea to read all the
details of the :download:`SPEAD <SPEAD_Protocol_Rev1_2012.pdf>` protocol, but
for the purposes of the tutorial we'll provide a brief overview here.

SPEAD is a message-based protocol, where the messages are called :dfn:`heaps`.
A sequence of heaps all sent to the same receiver is called a :dfn:`stream`.
Each heap contains a number of :dfn:`items`. Each item has

- a :dfn:`name`, which is a short string that can be used to look up the item.
  Typically it's a valid programming language identifier, but this is not
  required;
- an :dfn:`ID`, which is a small integer used to identify the item in the
  protocol. You can either assign your own IDs or you can let spead2 assign
  incremental IDs for you.
- a :dfn:`description`, which is a longer string meant to document the item
  for humans;
- a :dfn:`shape`, which indicates the dimensions of a multi-dimensional array.
  This can be empty for scalar values;
- a :dfn:`type`; and
- a :dfn:`value`.

It is quite common for a heap to contain only the ID and value of an item, to
avoid repeating all the other information if it does not change. The other
information is packaged into a special object called a :dfn:`descriptor` and
included into a heap. At a minimum, descriptors are sent in the first heap of
the stream, but may also be sent periodically (for the benefit of receivers
who missed the initial descriptors).

The Python bindings in spead2 contain utilities to manage all this. In
particular, an :dfn:`item group` holds all the items that are going to be
transmitted on a stream. It tracks which descriptors and values have already
been transmitted.  The general process is thus to modify some of the items,
request a heap from the item group, and transmit it on a stream.

The C++ bindings operate at a lower level. You will need to explicitly
construct each heap from the items you want, including any descriptors.

Sender
------
We'll start by writing an example transmitter, which means we need to know
what we're sending. We'll simulate a digitiser, which means having a
continuous stream of real values (representing voltages). Since SPEAD is
message-based, we'll need to split the values into chunks. Additionally,
we'll want to timestamp the data, which we'll do just with a sample counter.

In this section we'll build up the code a piece at a time, for both
Python and C++. You can see the whole thing in
:file:`examples/send_example.py` of the spead2 repository.

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
            boost::asio::ip::address::from_string("127.0.0.1"),
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

        chunk_size = 1024 * 1024
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
            shape=(chunk_size,),
            dtype=np.int8,
        )

 .. code-block:: c++
    :dedent: 0

        const std::int64_t chunk_size = 1024 * 1024;
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
            "{'shape': (" + std::to_string(chunk_size) + ",), 'fortran_order': False, 'descr': 'i1'}";

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
to ``timestamp``, with format ('i', 8) for 8-bit signed integer, but this
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
            item_group["timestamp"].value = i * chunk_size
            item_group["adc_samples"].value = rng.integers(-100, 100, size=chunk_size, dtype=np.int8)
            heap = item_group.get_heap()
            stream.send_heap(heap)

 .. code-block:: c++
    :dedent: 0

        std::mt19937 random_engine;
        std::uniform_int_distribution<std::int8_t> distribution(-100, 100);
        std::vector<std::int8_t> adc_samples(chunk_size);

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
            for (int i = 0; i < chunk_size; i++)
                adc_samples[i] = distribution(random_engine);
            // Add the data and timestamp to the heap
            heap.add_item(timestamp_desc.id, i * chunk_size);
            heap.add_item(
                adc_samples_desc.id,
                adc_samples.data(),
                adc_samples.size() * sizeof(adc_samples[0]),
                true
            );
            stream.async_send_heap(heap, boost::asio::use_future).wait();
        }

The Python code is reasonably straight-forward: we update the items, package
the changes into a heap, and transmit it. The C++ code needs more explanation.
Firstly, as mentioned earlier, the Python API takes care of sending
descriptors in the first heap, so that the receiver knows the names, shapes
and types of the items. In C++ we must explicitly add the descriptors on the
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
        stream.async_send_heap(heap, boost::asio::use_future).wait();
    }
