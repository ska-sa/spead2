Pipelining the sender
=====================
At present, our sender program only does one thing at a time, alternating
between generating some data (in our case, by just filling it with a
constant) and transmitting that data. Modern processors have multiple cores,
so we can speed things up by performing these operations in parallel. But
apart from just increasing performance for the sake of it, this is important
for real UDP applications because it allows the transmission to be smoothed out
with a constant transmission speed, rather than alternating idle periods with
rapid bursts. Rapid bursts can cause congestion with other traffic on the
network (leading to packet loss) or overwhelm receivers that don't have the
performance to absorb them.

.. tikz:: Serial computation (yellow) and transmission (green). Arrows show dependencies.
   :libs: positioning

   \tikzset{
     proc/.style={draw, minimum height=0.5cm},
     compute/.style={fill=yellow, proc, minimum width=0.5cm},
     transmit/.style={fill=green!50!white, proc, minimum width=0.7cm},
     label/.style={anchor=east},
     link/.style={->, shorten < = 0.05cm, shorten > = 0.05cm},
     >=latex,
   }
   \node[compute] (h1c) {};
   \node[transmit, right=0cm of h1c] (h1t) {};
   \node[compute, below=of h1t.east, anchor=west] (h2c) {};
   \node[transmit, right=0cm of h2c] (h2t) {};
   \node[compute, below=of h2t.east, anchor=west] (h3c) {};
   \node[transmit, right=0cm of h3c] (h3t) {};
   \node[label, left=of h1c] (h1l) {Heap 1};
   \node[label] (h2l) at (h1l.east |- h2c) {Heap 2};
   \node[label] (h3l) at (h1l.east |- h3c) {Heap 3};
   \coordinate[below=of h3l.east] (bstart);
   \coordinate[below=0.25 of bstart] (blow);
   \coordinate[above=0.25 of bstart] (bhigh);
   \draw (blow -| h1c.west)
         -| (bhigh -| h1t.west)
         -| (blow -| h1t.east)
         -| (bhigh -| h2t.west)
         -| (blow -| h2t.east)
         -| (bhigh -| h3t.west)
         -| (blow -| h3t.east)
         -- +(0.5, 0) coordinate (bstop);
   \node[label] (bwl) at (h1l.east |- bstart) {Bandwidth};
   \coordinate[below=of bwl.east] (tstart);
   \draw[->] (tstart) to[edge label'=Time] (tstart -| bstop);
   \draw[link] (h1t.south east) -- (h2c.north west);
   \draw[link] (h2t.south east) -- (h3c.north west);

.. tikz:: Parallel computation (yellow) and transmission (green).
   :libs: positioning

   \tikzset{
     proc/.style={draw, minimum height=0.5cm},
     compute/.style={fill=yellow, proc, minimum width=0.5cm},
     transmit/.style={fill=green!50!white, proc, minimum width=0.7cm},
     label/.style={anchor=east},
     link/.style={->, shorten < = 0.05cm, shorten > = 0.05cm},
     >=latex,
   }
   \node[compute] (h1c) {};
   \node[transmit, right=0cm of h1c] (h1t) {};
   \node[compute, below=of h1c.east, anchor=west] (h2c) {};
   \node[transmit, below=of h1t.east, anchor=west] (h2t) {};
   \node[compute, below=of h2c.east, anchor=west] (h3c) {};
   \node[transmit, below=of h2t.east, anchor=west] (h3t) {};
   \node[label, left=of h1c] (h1l) {Heap 1};
   \node[label] (h2l) at (h1l.east |- h2c) {Heap 2};
   \node[label] (h3l) at (h1l.east |- h3c) {Heap 3};
   \coordinate[below=of h3l.east] (bstart);
   \coordinate[below=0.25 of bstart] (blow);
   \coordinate[above=0.25 of bstart] (bhigh);
   \draw (blow -| h1c.west)
         -| (bhigh -| h1t.west)
         -| (blow -| h3t.east)
         -- +(0.5, 0) coordinate (bstop);
   \node[label] (bwl) at (h1l.east |- bstart) {Bandwidth};
   \coordinate[below=of bwl.east] (tstart);
   \draw[->] (tstart) to[edge label'=Time] (tstart -| bstop);
   \draw[link] (h1c.south east) -- (h2c.north west);
   \draw[link] (h2c.south east) -- (h3c.north west);
   \draw[link] (h1t.south east) -- (h2t.north west);
   \draw[link] (h2t.south east) -- (h3t.north west);


To generate and transmit data in parallel, we're going to need to restructure
our code a bit. In Python we'll use the :mod:`asyncio` module to manage
asynchronous sending, which means we'll need an asynchronous :py:func:`!main`
function. If you're using Python but you've never used :mod:`asyncio` before,
this would be a good time to find a tutorial on it, such as `this one`_.
Modify the code as follows:

.. _this one: https://realpython.com/async-io-python/

.. tab-set-code::

 .. code-block:: python

    import asyncio
    ...
    async def main():
        ...

    if __name__ == "__main__":
        asyncio.run(main())

We also need to use the asynchronous classes and methods of the spead2 API:

.. tab-set-code::

 .. code-block:: python

    import spead2.send.asyncio
    ...
        stream = spead2.send.asyncio.UdpStream(thread_pool, [(args.host, args.port)], config)
        ...
            await stream.async_send_heap(heap)
            ...
        await stream.async_send_heap(item_group.get_end())

That brings us to parity with the current C++ version, which already uses
``async_send_heap``. However, we haven't actually created any concurrency
yet, because immediately after starting the transmission, we wait for it to
complete (with ``await`` in Python or ``.get()`` in C++) before doing
anything else.

It's important to realise that ``async_send_heap`` does **not** necessarily
copy the heap data before transmitting it. Thus, between calling
``async_send_heap`` and waiting for it to complete, you must be careful not to
modify the data. If we are to prepare the next heap while the current heap is
being transmitted, we must do the preparation in different memory, and we
also need to ensure that the memory isn't freed while it is being used. We'll
use a :py:class:`!State` class to hold all the data that we need to associate
with a particular heap and keep alive until later. In Python this is simpler
because the garbage collector keeps things alive for us.

.. tab-set-code::

 .. code-block:: python

    from dataclasses import dataclass, field
    ...
    @dataclass
    class State:
        future: asyncio.Future[int] = field(default_factory=asyncio.Future)

 .. code-block:: c++

    struct state
    {
        std::future<spead2::item_pointer_t> future;
        std::vector<std::int8_t> adc_samples;
        spead2::send::heap heap;
    };

A "future" is an abstraction for a result that will only become available at
some point in the future, and on which one may wait; in this case the result
of transmitting a heap. If transmission fails, the result is an exception;
otherwise, it is the number of bytes actually transmitted (including
overheads from the SPEAD protocol, but excluding overheads from lower-level
protocols such as IP and UDP).

We're going to submit heap :math:`n+1` to ``async_send_heap`` while heap
:math:`n` is potentially still "in-flight". A stream has a bounded capacity
for in-flight heaps, which we can configure with the config object. The
default is actually more than 2, so this isn't necessary for our
example, but we'll be explicit in order to demonstrate the syntax.

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        config = spead2.send.StreamConfig(rate=0.0, max_heaps=2)

 .. code-block:: c++
    :dedent: 0

        config.set_max_heaps(2);

Now we rework the main loop to use the state class, and to delay retrieving
the result of the future for heap :math:`n` until we've passed heap
:math:`n+1` to ``async_send_heap``. Our diagram above isn't quite accurate,
because we don't start computing heap :math:`n+2` until we've retrieved the
result of heap :math:`n`. The actual situation is this (note the new arrow
from heap 1 to heap 3).

.. tikz:: Parallel computation (yellow) and transmission (green) with at most two heaps in flight.
   :libs: positioning

   \tikzset{
     proc/.style={draw, minimum height=0.5cm},
     compute/.style={fill=yellow, proc, minimum width=0.5cm},
     transmit/.style={fill=green!50!white, proc, minimum width=0.7cm},
     label/.style={anchor=east},
     link/.style={->, shorten < = 0.05cm, shorten > = 0.05cm},
     >=latex,
   }
   \node[compute] (h1c) {};
   \node[transmit, right=0cm of h1c] (h1t) {};
   \node[compute, below=of h1c.east, anchor=west] (h2c) {};
   \node[transmit, below=of h1t.east, anchor=west] (h2t) {};
   \node[compute, below=of h2t.west, anchor=west] (h3c) {};
   \node[transmit, below=of h2t.east, anchor=west] (h3t) {};
   \node[label, left=of h1c] (h1l) {Heap 1};
   \node[label] (h2l) at (h1l.east |- h2c) {Heap 2};
   \node[label] (h3l) at (h1l.east |- h3c) {Heap 3};
   \coordinate[below=of h3l.east] (bstart);
   \coordinate[below=0.25 of bstart] (blow);
   \coordinate[above=0.25 of bstart] (bhigh);
   \draw (blow -| h1c.west)
         -| (bhigh -| h1t.west)
         -| (blow -| h3t.east)
         -- +(0.5, 0) coordinate (bstop);
   \node[label] (bwl) at (h1l.east |- bstart) {Bandwidth};
   \coordinate[below=of bwl.east] (tstart);
   \draw[->] (tstart) to[edge label'=Time] (tstart -| bstop);
   \draw[link] (h1c.south east) -- (h2c.north west);
   \draw[link] (h2c.south east) to[bend right=40] (h3c.north west);
   \draw[link] (h1t.south east) -- (h2t.north west);
   \draw[link] (h2t.south east) -- (h3t.north west);
   \draw[link] (h1t.south east) to[bend right=15] (h3c.north west);

.. tab-set-code::

 .. code-block:: python
    :dedent: 0

        old_state = None
        start = time.perf_counter()
        for i in range(n_heaps):
            new_state = State()
            ...
            new_state.future = stream.async_send_heap(heap)
            if old_state is not None:
                await old_state.future
            old_state = new_state
        await old_state.future

 .. code-block:: c++
    :dedent: 0

    #include <memory>
    ...
        std::unique_ptr<state> old_state;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_heaps; i++)
        {
            auto new_state = std::make_unique<state>();
            auto &heap = new_state->heap;  // delete previous declaration of 'heap'
            auto &adc_samples = new_state->adc_samples;
            adc_samples.resize(heap_size, i);
            ...
            new_state->future = stream.async_send_heap(heap, boost::asio::use_future);
            if (old_state)
                old_state->future.get();
            old_state = std::move(new_state);
        }
        old_state->future.get();

Note how at the end of the loop we still need to wait for the final heap.

This improves performance to around 4000 MB/s for both Python and C++.

Apart from overlapping the data generation with the transmission,
there is another hidden benefit to this approach: pipelining. Even if the
data generation were free, the original code would have sub-optimal
performance because we wait until transmission is complete before submitting
the next batch of work. This means that the networking thread will go to sleep
after finishing heap :math:`n` and need to be woken up again when heap
:math:`n+1` is submitted, and no data is being transmitted while the thread is
being woken up. With the new code, provided the processing is fast enough to
submit heap :math:`n+1` before heap :math:`n` is complete, the worker thread
can move directly from one to the next without needing to pause. In our
example this makes no noticeable difference, but it can be significant if the
heaps are small, and it can even be beneficial to have more than two heaps in
flight at a time.

Full code
---------
.. tab-set-code::

   .. literalinclude:: ../../examples/tutorial/tut_5_send_pipeline.py
      :language: python

   .. literalinclude:: ../../examples/tutorial/tut_5_send_pipeline.cpp
      :language: c++
