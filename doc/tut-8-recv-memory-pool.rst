.. role:: pythoncode(code)
   :language: python

Memory pools
============

Sender
------

To demonstrate more features of spead2, we'll need to experiment with different
heap sizes. Instead of editing the hard-coded value, let's introduce another
command-line option. Don't forget to delete the original definition of
``heap_size``.

.. tab-set-code::

 .. code-block:: python

    async def main():
        ...
        parser.add_argument("-H", "--heap-size", type=int, default=1024 * 1024)
        ...
        heap_size = args.heap_size

 .. code-block:: c++

    static void usage(const char * name)
    {
        std::cerr << "Usage: " << name << " [-n heaps] [-p packet-size] [-H heap-size] host port\n";
    }

    int main(int argc, char * const argv[])
    {
        ...
        std::int64_t heap_size = 1024 * 1024;
        while ((opt = getopt(argc, argv, "n:p:H:")) != -1)
        {
            switch (opt)
            {
            ...
            case 'H':
                heap_size = std::stoll(optarg);
                break;
            ...
            }
        }

As with previous versions of the sender, the command-line parsing in C++ is
not very robust to user mistakes.

For this rest of this section we'll pass the options ``-n 100 -H 67108864`` to
use 64 MiB heaps (and reduce the number of heaps to speed things up). First
let us see what impact it has on the sender in isolation (this assumes you
have set up the dummy network interface as in :doc:`tut-4-send-perf`.

.. code-block:: sh

   tut_8_send -n 100 -H 67108864 192.168.31.2 8888

The performance is worse: significantly so in the C++ case (I get around 1750
MB/s for C++ and 5000 MB/s for Python). This is somewhat surprising, because
bigger heaps should mean that per-heap overheads are reduced, just like
increasing the packet size reduced the per-packet overheads. There are (at
least) two things going on here:

1. Caching. My CPU has a 1.5 MiB L2 cache (per core) and a 12 MiB L3 cache.
   The heap no longer fits into either of them, and so cache misses are
   substantially increased. In Python, the original command (1 MiB heaps)
   missed on 0.42% of L3 cache loads, while this new command misses on 6.4% of
   L3 cache loads.

2. Memory allocation. When the application allocates memory to hold the data
   for a heap, the underlying library can do it in one of two ways: it can
   either hand out some memory that it has previously requested from the
   operating system but which isn't in use, or it can request new memory from
   the operating system. In the latter case, Linux will provide a virtual
   memory address, but it won't actually allocate the physical memory.
   Instead, the first time each page is accessed, a page fault will occur, and
   the kernel will allocate a page of physical memory and zero it out. Page
   faults are expensive, so if this happens for every heap it becomes
   expensive.

   In Glibc (the standard C library on most Linux distributions) the memory
   allocator uses heuristics to try to avoid this. However, for allocations
   bigger than 32 MiB (at the time of writing) it will always request memory
   directly from the operating system, and return it directly to the operating
   system when it is freed. That is why we see such poor performance with our
   64 MiB heaps.

   In numpy the situation is slightly different: it is also obtaining the
   memory from the operating system, but it uses a hint to request that the
   memory is backed by "huge pages" (2 MiB pages on x86_64, compared to the
   default of 4 kiB pages). Since it takes far fewer pages to provide the
   physical memory, there are fewer page faults, and performance suffers less
   as a result.

We can't do anything about the caching problem [#cache-size-heaps]_, but we can
rewrite our code to avoid doing memory allocation on every iteration. We'll do
that by re-using our state class, but instead of creating a new one each
iteration, we'll keep a pool of two of them and alternate between them
(so-called "double-buffering").

In general when we start to fill in the data for a heap we need to make sure
that previous asynchronous use of that heap has completed (by waiting for a
corresponding future), but the first time each heap gets used is special. To
avoid having to deal with special cases, we can set things up with a future
that is already complete.

.. tab-set-code::

 .. code-block:: python

    @dataclass
    class State:
        adc_samples: np.ndarray
        future: asyncio.Future[int] = field(default_factory=asyncio.Future)

        def __post_init__(self):
            # Make it safe to wait on the future immediately
            self.future.set_result(0)

 .. code-block:: c++

    struct state
    {
        ...
        state()
        {
            // Make it safe to wait on the future immediately
            std::promise<spead2::item_pointer_t> promise;
            promise.set_value(0);
            future = promise.get_future();
        }
    };

Now we can get rid of ``old_state`` and ``new_state``, and instead use an
array of states.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        states = [State(adc_samples=np.ones(heap_size, np.int8)) for _ in range(2)]
        for i in range(n_heaps):
            state = states[i % len(states)]
            await state.future  # Wait for any previous use of this state to complete
            state.adc_samples.fill(i)
            item_group["timestamp"].value = i * heap_size
            item_group["adc_samples"].value = state.adc_samples
            heap = item_group.get_heap()
            state.future = stream.async_send_heap(heap)
        for state in states:
            await state.future

 .. code-block:: c++
    :dedent: 0

        std::array<state, 2> states;
        for (auto &state : states)
            state.adc_samples.resize(heap_size);
        for (int i = 0; i < n_heaps; i++)
        {
            auto &state = states[i % states.size()];
            // Wait for any previous use of this state to complete
            state.future.wait();
            auto &heap = state.heap;
            auto &adc_samples = state.adc_samples;

            heap = spead2::send::heap();  // reset to default state
            ...
            state.future = stream.async_send_heap(heap, boost::asio::use_future);
        }
        for (const auto &state : states)
            state.future.wait();

With this redesign, we now get close to 5000 MB/s from both C++ and Python.

.. [#cache-size-heaps] For this reason, it's generally a good idea to design
   your applications around a heap size that's small enough to fit into the L2
   cache.

Receiver
--------
The receiver (as implemented so far) suffers from the same problem of
repeatedly allocating memory from the OS and then incurring from page faults.
However, it differs from the sender in that the memory for heaps is allocated
by spead2 rather than the user. We thus need to use a spead2 feature to
address this: it allows us to set a custom allocator for heaps, and it also
provides one that recycles a pool of pre-allocated buffers.

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
            boost::asio::ip::address_v4::any(), std::atoi(argv[optind]));
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
  first two to 2 each. The C++ code only keeps one heap alive at a time, but
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
            0, args.heap_size + 8192, pool_heaps, pool_heaps
        )
        stream = spead2.recv.Stream(thread_pool, config, ring_config)

 .. code-block:: c++
    :dedent: 0

        spead2::recv::stream_config config;
        config.set_max_heaps(2);
        spead2::recv::ring_stream_config ring_config;
        ring_config.set_heaps(2);
        const int pool_heaps = config.get_max_heaps() + ring_config.get_heaps() + 1;
        config.set_memory_allocator(std::make_shared<spead2::memory_pool>(
            0, heap_size + 8192, pool_heaps, pool_heaps
        ));
        spead2::recv::ring_stream stream(thread_pool, config, ring_config);

With these changes, I'm reliably able to receive 64 MiB heaps across the
loopback interface.

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
still require memory to be allocated from the OS then faulted in, and can
cause initial heaps to be lost. The memory pool writes to its buffers when it
is constructed, which ensures that they are already paged in when the first
data is received. On the other hand, a memory pool adds some overhead, so
for very small heaps (hundreds of kB or less) you may get better performance
without one.

Full code
---------

Sender
^^^^^^

.. tab-set-code::

   .. literalinclude:: ../examples/tut_8_send.py
      :language: python

   .. literalinclude:: ../examples/tut_8_send.cpp
      :language: c++

Receiver
^^^^^^^^

.. tab-set-code::

   .. literalinclude:: ../examples/tut_8_recv.py
      :language: python

   .. literalinclude:: ../examples/tut_8_recv.cpp
      :language: c++
