Receiver, version 1
===================
Now that we have a sender, let's write a receiver. For the sake of an example,
let's have it report the average power (mean squared value) of the samples in
each heap. As before, the full code can be found at the bottom of the page, and
in the :file:`examples/tutorial` directory of the spead2 repository.

The initial boilerplate looks similar to the sender, and once again we'll need
a thread pool.

.. tab-set-code::

 .. code-block:: python

    #!/usr/bin/env python3

    import numpy as np
    import spead2.recv


    def main():
        thread_pool = spead2.ThreadPool()

 .. code-block:: c++

    #include <cassert>
    #include <cstdint>
    #include <cstddef>
    #include <iostream>
    #include <iomanip>
    #include <boost/asio.hpp>
    #include <spead2/common_ringbuffer.h>
    #include <spead2/common_thread_pool.h>
    #include <spead2/recv_ring_stream.h>
    #include <spead2/recv_udp.h>
    #include <spead2/recv_heap.h>

    int main()
    {
        spead2::thread_pool thread_pool;

Next we'll declare the stream. In C++ we declare it as a
:cpp:class:`~spead2::recv::ring_stream`, but in Python just as a
:py:class:`~spead2.recv.Stream`. They are the same concept, but the C++ API
provides other sorts of streams (hence the more verbose name) which the Python
API does not. The "ring" in the name indicates that incoming heaps are placed
on a ringbuffer. Readers can also take a configuration object (similarly to
senders), but we won't need one for now.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        stream = spead2.recv.Stream(thread_pool)

 .. code-block:: c++
    :dedent: 0

        spead2::recv::ring_stream stream(thread_pool);

You'll also notice that the class name does not specify the
underlying transport (UDP). Receive streams are quite flexible and in theory
can even accept packets from multiple different protocols simultaneously. To
feed data into a stream one must attach one or more :dfn:`readers`. In C++
each reader is represented by a class (in this case,
:cpp:class:`~spead2::recv::udp_reader`) although the user does not directly
instantiate the class. In Python there is an add method for each reader type.
We'll have our reader listen on UDP port 8888, the same port our sender is
transmitting on.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        stream.add_udp_reader(8888)

 .. code-block:: c++
    :dedent: 0

        boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::any(), 8888);
        stream.emplace_reader<spead2::recv::udp_reader>(endpoint);

Now we'll write a loop to iterate over the heaps. The processing of the heap
is left until later. For convenience, the stream object can be iterated to
obtain the heaps as they arrive.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        item_group = spead2.ItemGroup()
        for heap in stream:
            ...


    if __name__ == "__main__":
        main()

 .. code-block:: c++
    :dedent: 0

        for (const spead2::recv::heap &heap : stream)
        {
            ...
        }
        return 0;
    }

Now we'll fill in the body of the loop to process the heap, by computing the
mean of the squares of the samples.  In Python we can just update the item
group with the heap, which will create items from the descriptors in the first
heap and also update the values. The C++ API doesn't have item groups, and it leaves
interpretation of descriptors up to the user. Ideally we would parse the
descriptor to determine the item IDs for ``timestamp`` and ``adc_samples`` and
also learn about their types, but to keep things simple we'll just hard-code
our knowledge about them from the receiver. We're also hard-coding the
assumption that the timestamp has in fact been encoded as an immediate value,
for which spead2 provides a convenient way to retrieve it. If it wasn't
encoded as an immediate, we would have to use ``item.ptr`` and ``item.length``
to retrieve the raw 40-bit big-endian value and decode it.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

            item_group.update(heap)
            timestamp = item_group["timestamp"].value
            power = np.mean(np.square(item_group["adc_samples"].value, dtype=int))
            print(f"Timestamp: {timestamp:<10} Power: {power:.2f}")

 .. code-block:: c++
    :dedent: 0

            std::int64_t timestamp = -1;
            const std::int8_t *adc_samples = nullptr;
            std::size_t length = 0;
            for (const auto &item : heap.get_items())
            {
                if (item.id == 0x1600)
                {
                    assert(item.is_immediate);
                    timestamp = item.immediate_value;
                }
                else if (item.id == 0x3300)
                {
                    adc_samples = reinterpret_cast<const std::int8_t *>(item.ptr);
                    length = item.length;
                }
            }
            if (timestamp >= 0 && adc_samples != nullptr)
            {
                double power = 0.0;
                for (std::size_t i = 0; i < length; i++)
                    power += adc_samples[i] * adc_samples[i];
                power /= length;
                std::cout
                    << "Timestamp: " << std::setw(10) << std::left << timestamp
                    << " Power: " << power << '\n';
            }

Note that the Python code doesn't do any error checking: if we missed the
first heap, we won't receive the descriptors, and
so ``item_group["timestamp"]`` will raise a :exc:`KeyError`. You can test this
by starting the receiver slightly after the sender. Additionally,
:py:meth:`.ItemGroup.update` can fail for a number of reasons, such as a
transmitted item having the wrong number of bytes relative to its descriptor.

If you're following in C++, you'll again need to compile this example code
(see the previous section for instructions). Now run the receiver in one
terminal, then run the sender from the previous section in another. You should
see output something like the following:

.. code-block:: text

    Timestamp: 0          Power: 3328.61
    Timestamp: 1048576    Power: 3335.04
    Timestamp: 2097152    Power: 3330.53
    Timestamp: 3145728    Power: 3336.71
    Timestamp: 4194304    Power: 3333.94
    Timestamp: 5242880    Power: 3334.75
    Timestamp: 6291456    Power: 3336.29
    Timestamp: 7340032    Power: 3333.02
    Timestamp: 8388608    Power: 3334.64
    Timestamp: 9437184    Power: 3334.27

Full code
---------
.. tab-set-code::

   .. literalinclude:: ../../examples/tutorial/tut_3_recv.py
      :language: python

   .. literalinclude:: ../../examples/tutorial/tut_3_recv.cpp
      :language: c++
