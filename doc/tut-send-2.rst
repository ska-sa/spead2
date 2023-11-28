Sender, version 2
=================
Now that we have some functioning code, let's see how we can use more features
to improve performance. Of course, before trying to improve performance, we
ought to have some idea of what the performance is. We'll make some changes to
observe the time before and after we send the heaps, and also send more heaps
(to make the timing more reliable). We'll remove the target rate, so that
we're just sending as fast as we can. Setting the rate to 0 has the special
meaning of removing any rate limiting (it is also the default, so we could
just not set it at all).

The final code for this section can be found in :file:`examples/tut_send_2.py`
and :file:`examples/tut_send_2.cpp`. Unlike the previous sections though,
we'll be modifying the code as we go, rather than just writing it from top to
bottom.

.. tab-set-code::

 .. code-block:: python

    import time
    ...
        config = spead2.send.StreamConfig(rate=0.0)
        ...
        n_heaps = 100
        start = time.monotonic()
        for i in range(n_heaps):
            ...
        elapsed = time.monotonic() - start
        print(f"{chunk_size * n_heaps / elapsed / 1e6:.2f} MB/s")

 .. code-block:: c++

    #include <chrono>
    #include <iostream>
    #include <memory>  // Not needed yet, but we'll use it later
    ...
        config.set_rate(0.0);
        ...
        const int n_heaps = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_heaps; i++)
        {
            ...
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - start);
        std::cout << chunk_size * n_heaps / elapsed.count() / 1e6 << " MB/s\n";

You can expect performance to be pretty low; I get around 85 MB/s from Python
and 150 MB/s from C++.

There are multiple reasons for the low performance, but one of them is that
generating random numbers takes time, and we're alternating between sending a
heap and preparing the random numbers for the next heap. Modern processors
have multiple cores, so we can speed things up by performing these operations
in parallel. To do that, we're going to need to restructure our code a bit.

In Python we'll use the :mod:`asyncio` module to manage asynchronous sending,
which means we'll need an asynchronous :py:func:`!main` function. If you're
using Python but you've never used :mod:`asyncio` before, this would be a good
time to find a tutorial on it. Modify the code as follows:

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
        stream = spead2.send.asyncio.UdpStream(thread_pool, [("127.0.0.1", 8888)], config)
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
        for i in range(n_heaps):
            new_state = State()
            ...
            if old_state is not None:
                await old_state.future
            old_state = new_state
        await old_state.future

 .. code-block:: c++
    :dedent: 0

        std::unique_ptr<state> old_state;
        for (int i = 0; i < n_heaps; i++)
        {
            auto new_state = std::make_unique<state>();
            auto &heap = new_state->heap;
            auto &adc_samples = new_state->adc_samples;
            adc_samples.resize(chunk_size);
            ...
            new_state->future = stream.async_send_heap(heap, boost::asio::use_future);
            if (old_state)
                old_state->future.get();
            old_state = std::move(new_state);
        }
        old_state->future.get();

Note how at the end of the loop we still need to wait for the final heap.

This improves performance to around 100 MB/s for Python and 250 MB/s for C++.
The Python code does not get much speedup because the random number generation
is a bottleneck, but the C++ code is now significantly faster.

Apart from overlapping the random number generation with the transmission,
there is another hidden benefit to this approach: pipelining. Even if the
random number generation were free, the original code would have sub-optimal
performance because we wait until transmission is complete before submitting
the next batch of work. This means that the networking thread will go to sleep
after finishing heap :math:`n` and need to be woken up again when heap
:math:`n+1` is submitted, and no data is being transmitted during that time.
With the new code, provided the processing is fast enough to submit heap
:math:`n+1` because heap :math:`n` is complete, the worker thread can move
directly from one to the next without needing to pause. In our example this
makes no noticeable difference, but it can be significant if the heaps are
small, and it can even be beneficial to have more than two heaps in flight at
a time.
