.. role:: pythoncode(code)
   :language: python

Memory pools
============
In the previous section, we optimised the sender by ensuring memory did not get
repeatedly allocated from the OS then returned. The receiver suffers from the
same problem. However, it differs from the sender in that the memory for heaps
is allocated by spead2 rather than the user. We thus need to use a spead2
feature to address this: it allows us to set a custom allocator for heaps, and
it also provides one that recycles a pool of pre-allocated buffers.

To use this custom allocator, we need to know how much memory to allocate up
front, before receiving any packets. We'll add command-line argument parsing
to the receiver to facilitate this; while we're at it, we'll make the port
number a command-line argument instead of being hard-coded to 8888.

.. tab-set-code::

 .. code-block:: python

    import argparse
    ...
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("-H", "--heap-size", type=int, default=1024 * 1024)
        parser.add_argument("port", type=int)
        args = parser.parse_args()
        ...
        stream.add_udp_reader(args.port)

 .. code-block:: c++

    #include <unistd.h>
    #include <string>
    ...
    static void usage(const char *name)
    {
        std::cerr << "Usage: " << name << " [-H heap-size] port\n";
    }

    int main(int argc, char * const argv[])
    {
        int opt;
        std::int64_t heap_size = 1024 * 1024;
        while ((opt = getopt(argc, argv, "H:")) != -1)
        {
            switch (opt)
            {
            case 'H':
                heap_size = std::stoll(optarg);
                break;
            default:
                usage(argv[0]);
                return 2;
            }
        }
        if (argc - optind != 1)
        {
            usage(argv[0]);
            return 2;
        }

        ...
        boost::asio::ip::udp::endpoint endpoint(
            boost::asio::ip::address_v4::any(), std::stoi(argv[optind]));
        ...
    }


Now we need to actually create and use the memory pool. When creating a memory
pool, we need to specify a few parameters:

- The minimum heap payload size for which we will use the pool. We can just set
  this to zero to use the pool all the time. If you're expecting to have a mix
  of large and tiny heaps (the latter might contain only descriptors, for
  example), it may be worth setting a non-zero value for this so that the tiny
  heaps don't consume from the memory pool.
- The maximum heap payload size for which we will use the pool. This determines
  how much memory is actually allocated for each buffer in the pool. This can
  be a little tricky: we know exactly how much space is needed for the actual
  data, but the payload can also contain things like the descriptors sent with
  the first heap. We'll just play it safe and allocate an extra 8192 bytes,
  which just means we'll use slightly more memory than absolutely necessary.
- The number of buffers to allocate. This is also tricky to get right if we
  want to avoid allocating new memory in the middle of receiving data. There
  are three places that memory might need to be allocated: incomplete heaps
  that spead2 is still receiving data for, complete heaps in the ringbuffer,
  and heaps that your code has received and not yet deleted. We'll limit the
  first two to two each. The C++ code only keeps one heap alive at a time, but
  the Python version actually holds references to two: until the call to
  :pythoncode:`item_group.update`, the item group still references the data
  from the previous heap. It should also be noted that while CPython frees
  heaps as soon as they're no longer referenced, PyPy (and any other
  Python implementation that doesn't use reference counting) might cause heaps
  to linger for an unknown amount of time. PyPy is thus not recommended for
  use with memory pools, and in general is not recommended for spead2
  receivers due to unpredictable performance.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        config = spead2.recv.StreamConfig(max_heaps=2)
        ring_config = spead2.recv.RingStreamConfig(heaps=2)
        pool_heaps = config.max_heaps + ring_config.heaps + 2
        config.memory_allocator = spead2.MemoryPool(
            lower=0,
            upper=args.heap_size + 8192,
            max_free=pool_heaps,
            initial=pool_heaps,
        )
        stream = spead2.recv.Stream(thread_pool, config, ring_config)

 .. code-block:: c++
    :dedent: 0

    #include <spead2/common_memory_pool.h>
    ...

        spead2::recv::stream_config config;
        config.set_max_heaps(2);
        spead2::recv::ring_stream_config ring_config;
        ring_config.set_heaps(2);
        const int pool_heaps = config.get_max_heaps() + ring_config.get_heaps() + 1;
        config.set_memory_allocator(std::make_shared<spead2::memory_pool>(
            0,                 // lower
            heap_size + 8192,  // upper
            pool_heaps,        // max_free
            pool_heaps         // initial
        ));
        spead2::recv::ring_stream stream(thread_pool, config, ring_config);

With these changes, I'm able to receive 64 MiB heaps across the
loopback interface most of the time, using the following commands:

.. code-block:: sh

   tut_9_recv_memory_pool -H 67108864 8888
   tut_8_send_reuse_memory -n 100 -H 67108864 -p 9000 127.0.0.1 8888

If you set the number of buffers too low and your memory pool becomes empty,
you'll get a warning (``memory pool is empty when allocating 67108864
bytes``). However, you might not encounter the worst case while testing, so
you shouldn't interpret the lack of such a warning to mean that you've sized
your memory pool correctly. If you can afford the extra memory usage, it's
often best to allocate slightly more than you think you need, just to be
safe.

Even when the heap size is small enough for the libc memory allocator to
retain and reuse buffers for heaps, using a spead2 memory pool can be
beneficial to the startup performance. Without it, the first few heaps will
still require memory to be allocated from the OS then paged in, and can
cause initial heaps to be lost. The memory pool writes to its buffers when it
is constructed, which ensures that they are already paged in when the first
data is received. On the other hand, a memory pool adds some overhead, so
for very small heaps (hundreds of kB or less) you may get better performance
without one.

Full code
---------
.. tab-set-code::

   .. literalinclude:: ../../examples/tutorial/tut_9_recv_memory_pool.py
      :language: python

   .. literalinclude:: ../../examples/tutorial/tut_9_recv_memory_pool.cpp
      :language: c++
