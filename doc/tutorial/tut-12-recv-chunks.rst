Receiving chunks
================

The previous two sections vastly improved transmit performance for small
heaps. We'll now turn to improving the receiver performance for small heaps.
The APIs we'll use can actually be useful for more than just very small heaps
though; they're generally useful when you want to group the payload for
multiple heaps into a larger contiguous array, or when you want more control
over the memory allocation and management.

The basic approach is similar to the batching used in the previous section,
but batching on the receive side is more complicated:

1. UDP is unreliable, so we need to be able to handle batches that are missing
   some heaps.

2. We don't know in what order heaps will be received, so we need to be able
   to use metadata (in our application, the timestamp) to steer heaps to the
   right place.

This steering is done by a callback function that we provide to spead2. This
poses a challenge for Python: we saw in the previous section that interacting
with the Python interpreter on a per-heap basis is bad for performance, but
this callback function needs to be called for each heap. Additionally, the
function is called from the thread pool rather than the main thread, and if it
tries to execute Python code there will be contention for the Global
Interpreter Lock (GIL). To avoid these problems, we need a function that
doesn't interact with Python at all. We could write it in C (or another
compiled language like C++ or Rust), but numba_ conveniently provides a way to
write C-compatible functions right inside our Python code using Python syntax.
In the C++ version of the tutorial we'll just have a normal C++ function.

.. _numba: http://numba.org/

This is going to be a major rewrite of the code, so I'll present it from
scratch rather than as edits to previous versions.

.. tab-set-code::

 .. code-block:: python

    #!/usr/bin/env python3

    import argparse
    import ctypes

    import numba
    import numpy as np
    import scipy
    from numba import types

    import spead2.recv
    from spead2.numba import intp_to_voidptr
    from spead2.recv.numba import chunk_place_data

 .. code-block:: c++

    #include <cstdint>
    #include <cstddef>
    #include <utility>
    #include <string>
    #include <iostream>
    #include <iomanip>
    #include <algorithm>
    #include <memory>
    #include <numeric>
    #include <unistd.h>
    #include <boost/asio.hpp>
    #include <spead2/common_defines.h>
    #include <spead2/common_ringbuffer.h>
    #include <spead2/common_thread_pool.h>
    #include <spead2/recv_stream.h>
    #include <spead2/recv_chunk_stream.h>
    #include <spead2/recv_udp.h>

There are some familiar imports/includes but also some new ones. In Python
there are a number of imports related to numba and scipy, which are providing
the utilities we'll use to write a C ABI function.

Next, we'll have a modified version of the function for computing mean power.
It will take a whole chunk as input, along with an array indicating whether
each heap of the chunk was received or not, and return the average power for
the whole chunk (excluding missing data). We won't try to report per-heap power
because printing that information would slow down the receiver too much.

.. tab-set-code::

 .. code-block:: python

    @numba.njit
    def mean_power(adc_samples, present):
        total = np.int64(0)
        n = 0
        for i in range(len(present)):
            if present[i]:
                for j in range(adc_samples.shape[1]):
                    sample = np.int64(adc_samples[i, j])
                    total += sample * sample
                n += adc_samples.shape[1]
        return np.float64(total) / n

 .. code-block:: c++

    #if defined(__GNUC__) && defined(__x86_64__)
    // Compile this function with AVX2 for better performance. Remove this if your
    // CPU does not support AVX2 (e.g., if you get an Illegal Instruction error).
    [[gnu::target("avx2")]]
    #endif
    static double mean_power(const std::int8_t *adc_samples, const std::uint8_t *present,
                             std::size_t heap_size, std::size_t heaps)
    {
        std::int64_t sum = 0;
        std::size_t n = 0;
        for (std::size_t i = 0; i < heaps; i++)
        {
            if (present[i])
            {
                for (std::size_t j = 0; j < heap_size; j++)
                {
                    std::int64_t sample = adc_samples[i * heap_size + j];
                    sum += sample * sample;
                }
                n += heap_size;
            }
        }
        return double(sum) / n;
    }

Now we come to the key component: the placement function that indicates what
to do with each heap. It receives a pointer to a
:cpp:struct:`~spead2::recv::chunk_place_data` structure, which contains input metadata
about the heap, as well as output fields that the function should write. One
of the input fields is ``items``, which contains the values of immediate items
that our code requests. Later on we'll request that this contains the heap
size and the timestamp.

.. tab-set-code::

 .. code-block:: python

    @numba.cfunc(
        types.void(types.CPointer(chunk_place_data), types.size_t, types.CPointer(types.int64)),
        nopython=True,
    )
    def place_callback(data_ptr, data_size, sizes_ptr):
        data = numba.carray(data_ptr, 1)
        items = numba.carray(intp_to_voidptr(data[0].items), 2, dtype=np.int64)
        sizes = numba.carray(sizes_ptr, 2)
        payload_size = items[0]
        timestamp = items[1]
        heap_size = sizes[0]
        chunk_size = sizes[1]

 .. code-block:: c++

    void place_callback(
        spead2::recv::chunk_place_data *data,
        std::int64_t heap_size, std::int64_t chunk_size)
    {
        auto payload_size = data->items[0];
        auto timestamp = data->items[1];

For once the Python version is more complicated, because it is interfacing
between different language paradigms. It is worth reading the
:external+numba:doc:`numba cfunc <user/cfunc>`
documentation to better understand it. The second parameter is the size of the
structure being pointed to. This is to allow code to be compatible with
multiple versions of spead2, where some fields might only exist in newer
versions. We're not depending on any fields that didn't exist from the start,
so we can ignore it. The third parameter we get to supply ourselves, but it
can only be a pointer. We actually want to pass two integers (the expected
heap size and chunk size), so we pass a pointer to an array of two integers.

Before placing a heap, we should check that it is actually suitable: it should
have a timestamp item, and it should be the right size (otherwise we might
overflow the allocated memory and crash!) If an immediate item is missing, it
will be reported as ``-1`` in this function.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        if timestamp >= 0 and payload_size == heap_size:

 .. code-block:: c++
    :dedent: 0

        if (timestamp >= 0 && payload_size == heap_size)
        {

Ok, we've got a valid heap. We now need to tell spead2 three things:

1. Which *chunk* does this heap belong to. Chunks should be numbered
   sequentially, so we'll assign chunk *i* to the time interval
   [*i* × chunk-size, (*i* + 1) × chunk-size).

2. At what byte offset within the chunk should the payload for this heap be
   written.

3. Which number heap is this of the chunk. This is used solely to set the flag
   indicating that the heap was successfully received. We can choose to number
   the heaps in a chunk however we like (even discontiguously), provided we
   allocate the ``present`` array with enough space. But we'll keep things
   simple, and number the heaps in the chunk in timestamp order.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

            data[0].chunk_id = timestamp // chunk_size
            data[0].heap_offset = timestamp % chunk_size
            data[0].heap_index = data[0].heap_offset // heap_size

 .. code-block:: c++
    :dedent: 0

            data->chunk_id = timestamp / chunk_size;
            data->heap_offset = timestamp % chunk_size;
            data->heap_index = data->heap_offset / heap_size;
        }
    }

Now we get to the main function. The command-line parsing is unchanged:

.. tab-set-code::

 .. code-block:: python

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("-H", "--heap-size", type=int, default=1024 * 1024)
        parser.add_argument("port", type=int)
        args = parser.parse_args()

 .. code-block:: c++

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

We need to decide how big to make the chunks. As with the batch size in the
previous tutorial, we want chunks to have the same order of magnitude as the
L2 cache. We'll aim for 1 MiB, but adjust it to be a multiple of the given
heap size.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        heap_size = args.heap_size
        chunk_size = 1024 * 1024  # Preliminary value
        chunk_heaps = max(1, chunk_size // heap_size)
        chunk_size = chunk_heaps * heap_size  # Final value

 .. code-block:: c++
    :dedent: 0

        std::int64_t chunk_size = 1024 * 1024;  // Preliminary value
        std::int64_t chunk_heaps = std::max(std::int64_t(1), chunk_size / heap_size);
        chunk_size = chunk_heaps * heap_size;  // Final value

Now we create the thread pool and stream config object. We'll pin the threads
to CPU cores 2 and 3 to get more reliable performance, just as the sender is
pinned to cores 0 and 1.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        thread_pool = spead2.ThreadPool(1, [2])
        spead2.ThreadPool.set_affinity(3)
        config = spead2.recv.StreamConfig(max_heaps=2)

 .. code-block:: c++
    :dedent: 0

        spead2::thread_pool thread_pool(1, {2});
        spead2::thread_pool::set_affinity(3);
        spead2::recv::stream_config config;
        config.set_max_heaps(2);

Next, we create another configuration object describing how the chunking is
done. This is where we
indicate the immediate items that we want made available in
the ``items`` array in ``place_callback``, namely the heap length and the
timestamp. Notice that we've specified the timestamp by ID (0x1600): this
interface does not support dynamically learning the ID from the descriptors,
and in fact this program will not depend on the descriptors at all.
We also specify the maximum number of chunks that can be under construction at
once. For this tutorial we're not expecting to receive data out of order, so we'll
just keep one in flight. In other words, as soon as we see a heap for a given
chunk, we'll assume all previous chunks are as complete as they'll ever be and
start processing them. Finally, we pass in the ``place_callback`` function. In
the Python code, we have to create the array of two integers whose pointer we
pass, as described earlier. In C++, we capture them using a lambda.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        user_data = np.array([heap_size, chunk_size], np.int64)
        chunk_config = spead2.recv.ChunkStreamConfig(
            items=[spead2.HEAP_LENGTH_ID, 0x1600],
            max_chunks=1,
            place=scipy.LowLevelCallable(
                place_callback.ctypes,
                user_data.ctypes.data_as(ctypes.c_void_p),
                "void (void *, size_t, void *)",
            ),
        )

 .. code-block:: c++
    :dedent: 0

        spead2::recv::chunk_stream_config chunk_config;
        chunk_config.set_items({spead2::HEAP_LENGTH_ID, 0x1600});
        chunk_config.set_max_chunks(1);
        chunk_config.set_place(
            [=](auto data, auto) { place_callback(data, heap_size, chunk_size); }
        );

The old receiver code used a ringbuffer to pass heaps from spead2 to our
application, but it was managed internally by the stream. The chunking API is
newer and more flexible, and separates the ringbuffer from the stream to allow
it to be shared between streams. It also uses a second ringbuffer to carry
free chunks from the application back to the stream. This replaces the memory
pool: instead of chunks being implicitly returned to a pool when they're
freed, we must explicitly put them onto this ringbuffer. In Python this has
the advantage that one controls exactly when this happens rather than needing
to rely on the garbage collector.

.. tikz:: Data flow with the chunk API.
   :libs: positioning,fit

   \tikzset{
     >=latex,
     every label/.style={font=\scriptsize},
     elabel/.style={auto, font=\scriptsize},
     nlabel/.style={font=\small},
   }
   \begin{scope}[shift={(0, 0)}]
     \fill[gray!50!white] (0, 0.5) +(45:0.2) arc[start angle=45, end angle=180, radius=0.2]
       -- (-0.5, 0.5) arc[start angle=180, end angle=45, radius=0.5]
       -- cycle;
     \draw (0, 0.5) circle (0.5);
     \draw (0, 0.5) circle (0.2);
     \foreach \i in {0, 45, ..., 315} {
       \draw (0, 0.5) +(\i:0.2) -- +(\i:0.5);
     }
     \node[fit={(-0.5, 0.5) (0.5, 0.5) (0, 1) (0, 0)}, label=left:Data ring] (data) {};
   \end{scope}

   \begin{scope}[shift={(4, 0)}]
     \fill[gray!50!white] (0, 0.5) +(45:0.2) arc[start angle=45, end angle=180, radius=0.2]
       -- (-0.5, 0.5) arc[start angle=180, end angle=45, radius=0.5]
       -- cycle;
     \draw (0, 0.5) circle (0.5);
     \draw (0, 0.5) circle (0.2);
     \foreach \i in {0, 45, ..., 315} {
       \draw (0, 0.5) +(\i:0.2) -- +(\i:0.5);
     }
     \node[fit={(-0.5, 0.5) (0.5, 0.5) (0, 1) (0, 0)}, label=right:Free ring] (free) {};
   \end{scope}

   \begin{scope}[shift={(1, 1.5)}]
     \draw (0, 0) rectangle (2, 1);
     \node[fit={(0, 0) (2, 1)}] (user) {};
     \node[text width=1.5cm, align=center, nlabel] at (user.center) {User thread};
   \end{scope}

   \begin{scope}[shift={(1, -1.5)}]
     \draw (0, 0) rectangle (2, 1);
     \node[fit={(0, 0) (2, 1)}] (worker) {};
     \node[text width=2cm, align=center, nlabel] at (worker.center) {Worker thread};
   \end{scope}

   \draw[->] (data) |- node[elabel] {Chunks} (user);
   \draw[->] (user) -| node[pos=0.49,elabel] {Chunks} (free);
   \draw[->] (free) |- node[elabel] {Chunks} (worker);
   \draw[->] (worker) -| node[elabel,pos=0.49] {Chunks} (data);

The data ringbuffer is kept small — we just need enough capacity to avoid
stalling the producer if the consumer is temporarily a little too slow. We'll
discuss the sizing of the free ringbuffer later.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        data_ring = spead2.recv.ChunkRingbuffer(2)
        free_ring = spead2.recv.ChunkRingbuffer(4)
        stream = spead2.recv.ChunkRingStream(
            thread_pool, config, chunk_config, data_ring, free_ring
        )

 .. code-block:: c++
    :dedent: 0

        using ringbuffer = spead2::ringbuffer<std::unique_ptr<spead2::recv::chunk>>;
        auto data_ring = std::make_shared<ringbuffer>(2);
        auto free_ring = std::make_shared<ringbuffer>(4);
        spead2::recv::chunk_ring_stream stream(
            thread_pool, config, chunk_config, data_ring, free_ring
        );

Now we'll create the actual chunks. Unlike with the memory pool, we are
responsible for allocating the memory. In C++, it is also required to store
the size of the ``present`` array (in Python it is taken from the size of the
buffer object).

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        for _ in range(free_ring.maxsize):
            chunk = spead2.recv.Chunk(
                data=np.zeros((chunk_heaps, heap_size), np.int8),
                present=np.zeros(chunk_heaps, np.uint8),
            )
            stream.add_free_chunk(chunk)

 .. code-block:: c++
    :dedent: 0

        for (std::size_t i = 0; i < free_ring->capacity(); i++)
        {
            auto chunk = std::make_unique<spead2::recv::chunk>();
            chunk->present = std::make_unique<std::uint8_t[]>(chunk_heaps);
            chunk->present_size = chunk_heaps;
            chunk->data = std::make_unique<std::uint8_t[]>(chunk_size);
            stream.add_free_chunk(std::move(chunk));
        }

The call to ``add_free_chunk`` places the new chunk onto the free ring, while
also zeroing out the ``present`` array (you can directly place chunks onto the
free ring yourself, but then you **must** do this zeroing out yourself).

We have created the same number of chunks as there is capacity in the free
ring. There is no need to make the free ring bigger (as it cannot contain more
chunks than are in existence), and if we made it smaller then we'd fail to add
all the chunks to it here. But how did we come up with the size of 4? It is
similar to the calculation for the capacity of the memory pool in section 9.
We need to have enough chunks for those under construction (1), those waiting
in the data ringbuffer (2) and those being processed by the application (1).

That concludes the setup, other than adding the reader, after which we're
ready for the main processing loop.  This looks reasonably similar to what we
had before, with the difference being that we're now processing chunks instead
of heaps. The timestamp we're reporting is the timestamp for the first heap in
the chunk.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        stream.add_udp_reader(args.port)
        n_heaps = 0
        # Run it once to trigger compilation for int8
        mean_power(np.ones((1, 1), np.int8), np.ones(1, np.uint8))
        for chunk in data_ring:
            timestamp = chunk.chunk_id * chunk_size
            n = int(np.sum(chunk.present, dtype=np.int64))
            if n > 0:
                power = mean_power(chunk.data, chunk.present)
                n_heaps += n
                print(f"Timestamp: {timestamp:<10} Power: {power:.2f}")
            stream.add_free_chunk(chunk)
        print(f"Received {n_heaps} heaps")

 .. code-block:: c++
    :dedent: 0

        boost::asio::ip::udp::endpoint endpoint(
            boost::asio::ip::address_v4::any(), std::stoi(argv[optind]));
        stream.emplace_reader<spead2::recv::udp_reader>(endpoint);
        std::int64_t n_heaps = 0;
        for (std::unique_ptr<spead2::recv::chunk> chunk : *data_ring)
        {
            auto present = chunk->present.get();
            auto n = std::accumulate(present, present + chunk_heaps, std::size_t(0));
            if (n > 0)
            {
                std::int64_t timestamp = chunk->chunk_id * chunk_size;
                auto adc_samples = (const std::int8_t *) chunk->data.get();
                n_heaps += n;
                double power = mean_power(adc_samples, present, heap_size, chunk_heaps);
                std::cout
                    << "Timestamp: " << std::setw(10) << std::left << timestamp
                    << " Power: " << power << '\n';
            }
            stream.add_free_chunk(std::move(chunk));
        }
        std::cout << "Received " << n_heaps << " heaps\n";
        return 0;
    }

Ok, let's give it a try. Run the following in two terminals:

.. code-block:: sh

   tut_12_recv_chunks -H 8192 8888
   tut_11_send_batch_heaps -n 524288 -H 8192 -p 9000 127.0.0.1 8888

If all goes well, you should see a lot of output of timestamp and power,
ending with something like this:

.. code-block:: text

    Timestamp: 4290772992 Power: 5397.50
    Timestamp: 4291821568 Power: 5525.50
    Timestamp: 4292870144 Power: 5397.50
    Timestamp: 4293918720 Power: 5525.50
    Received 524287 heaps

Wait a second, we sent 524288 heaps (not counting the heap that just contains
the end-of-stream notification), so we're missing one! It's the first heap: it
contains descriptors, which form part of the heap payload. Our
``place_callback`` rejects heaps that don't have the right payload size, so it
gets dropped. If you'd like an additional challenge, modify the sender to fix
this. One solution is to send the descriptors in a heap of their own, instead
of as part of the first data heap.

This is one demonstration that while this new receiver is much faster for
small heaps, it is also much more brittle. It will only work correctly if the
incoming heaps are formatted in just the right way:

- The timestamp must have ID 0x1600.
- The timestamp must be an immediate item.
- The timestamps must be aligned to the heap size.
- The number of ADC samples must match the :option:`!-H` command-line option.
- There cannot be any other non-immediate items.

It's thus recommended to prefer larger heaps when possible.

Full code
---------
.. tab-set-code::

   .. literalinclude:: ../../examples/tutorial/tut_12_recv_chunks.py
      :language: python

   .. literalinclude:: ../../examples/tutorial/tut_12_recv_chunks.cpp
      :language: c++
