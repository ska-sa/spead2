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

.. TODO: insert a diagram here

To generate and transmit data in parallel, we're going to need to restructure
our code a bit. In Python we'll use the :mod:`asyncio` module to manage
asynchronous sending, which means we'll need an asynchronous :py:func:`!main`
function. If you're using Python but you've never used :mod:`asyncio` before,
this would be a good time to find a tutorial on it. Modify the code as
follows:

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
:math:`n+1` to ``async_send_heap``.

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
