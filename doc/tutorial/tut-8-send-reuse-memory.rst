Reusing memory
==============

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

For this rest of this section we'll pass the options ``-n 100 -H 67108864 -p 9000`` to
use 64 MiB heaps (and reduce the number of heaps to speed things up). First
let us see what impact it has on the sender in isolation (this assumes you
have set up the dummy network interface as in :doc:`tut-4-send-perf`).

.. code-block:: sh

   tut_8_send_reuse_memory -n 100 -H 67108864 -p 9000 192.168.31.2 8888

The performance is worse: significantly so for C++ (I get around 1300
MB/s) and slightly for Python (5900 MB/s). This is somewhat surprising, because
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
iteration, we'll keep a pool of two and alternate between them
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

    #include <future>
    ...
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
        start = time.perf_counter()
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

    #include <array>
    ...
        std::array<state, 2> states;
        for (auto &state : states)
            state.adc_samples.resize(heap_size);
        auto start = std::chrono::high_resolution_clock::now();
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

With this redesign, we now get around 5600 MB/s from C++ and 6000 MB/s from
Python (the difference is most likely due to Python using huge pages).

.. [#cache-size-heaps] For this reason, it's generally a good idea to design
   your applications around a heap size that's small enough to fit into the L2
   cache.

Full code
---------

.. tab-set-code::

   .. literalinclude:: ../../examples/tutorial/tut_8_send_reuse_memory.py
      :language: python

   .. literalinclude:: ../../examples/tutorial/tut_8_send_reuse_memory.cpp
      :language: c++
