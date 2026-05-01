Optimising the power calculation
================================
Now that we've made significant optimisations to the sender, let's turn our
attention back to the receiver. As with the sender, the first step is to
measure the performance. That is surprisingly tricky to do when
receiving a lossy protocol such as UDP, because the transmission rate is
determined by the sender, and all you get to measure is whether the receiver
was able to keep up (and if not, how much data was lost). What one really
wants to answer is "at what rate can the receiver reliably keep up", and
that's affected not just by the average throughput, but also random
variation.

Nevertheless, let's see if we can get our receiver to keep up with our sender.
Firstly, let's see what happens if we run it as is: run the receiver in one
terminal, and the sender (with arguments ``-p 9000 127.0.0.1 8888``) in
another. You will probably see a number of messages like the following:

.. code-block:: text

    dropped incomplete heap 20 (367344/1048576 bytes of payload)
    worker thread blocked by full ringbuffer on heap 81

The first one tells you that data has been lost. Recall that the sender is
dividing each heap into packets, each containing about 9000 bytes out of the 1 MiB
of sample data. For this particular heap, we received only
367344 bytes of payload, and so it was dropped. It is worth mentioning that
the heap ID (20) is one assigned by spead2 in the sender and encoded into
the packet; it does not necessarily match the loop index in the ``for`` loop
in the sender code.

.. tikz:: Data flow in the receive path.
   :libs: positioning,fit

   \tikzset{
     >=latex,
     every label/.style={font=\scriptsize},
     elabel/.style={auto, font=\scriptsize},
     nlabel/.style={font=\small},
   }
   \begin{scope}[xshift=0]
     \draw (0.1, -0.1) -- (1, -0.1) -- (1, 0) -- (2.5, 0) -- (2.5, 1) -- (0, 1) -- (0, 0) -| cycle;
     \node[fit={(0, 0) (2.5, 1)}, nlabel] (nic) {NIC};
   \end{scope}

   \begin{scope}[shift={(4, 0)}]
     \draw[shift={(0, 0.2)}] (0, 0) grid[xstep=0.25, ystep=0.6] (2, 0.6);
     \node[fit={(0, 0) (2, 1)}, label=below:Socket] (socket) {};
   \end{scope}

   \begin{scope}[shift={(8, 0)}]
     \draw (0, 0) rectangle (2, 1);
     \node[fit={(0, 0) (2, 1)}] (worker) {};
     \node[text width=2cm, align=center, nlabel] at (worker.center) {Worker thread};
   \end{scope}

   \begin{scope}[shift={(12, 0)}]
     \fill[gray!50!white] (0, 0.5) +(45:0.2) arc[start angle=45, end angle=180, radius=0.2]
       -- (-0.5, 0.5) arc[start angle=180, end angle=45, radius=0.5]
       -- cycle;
     \draw (0, 0.5) circle (0.5);
     \draw (0, 0.5) circle (0.2);
     \foreach \i in {0, 45, ..., 315} {
       \draw (0, 0.5) +(\i:0.2) -- +(\i:0.5);
     }
     \node[fit={(-0.5, 0.5) (0.5, 0.5) (0, 1) (0, 0)},
     label=below:Ringbuffer] (ringbuffer) {};
   \end{scope}

   \begin{scope}[shift={(14, 0)}]
     \draw (0, 0) rectangle (2, 1);
     \node[fit={(0, 0) (2, 1)}] (user) {};
     \node[text width=1.5cm, align=center, nlabel] at (user.center) {Main thread};
   \end{scope}

   \draw[->] (nic) -- node[elabel] {Packets} (socket);
   \draw[->] (socket) -- node[elabel] {Packets} (worker);
   \draw[->] (worker) -- node[elabel] {Heaps} (ringbuffer);
   \draw[->] (ringbuffer) -- node[elabel] {Heaps} (user);

The "full ringbuffer" message is more of a warning. The worker thread (in the
thread pool) operates by receiving data from the network socket and then
pushing complete heaps into a ringbuffer. If our code doesn't take those heaps
off the ringbuffer fast enough, then the ringbuffer fills up, and the worker
thread has to wait for space before it can push the next heap. While it's
waiting, it's not consuming packets from the socket, which eventually fills up
and causes subsequent packets to be dropped. On its own, the "full ringbuffer"
message isn't necessarily an issue, but when we see it together with lost data,
it tells us that it's the processing in our own code that's too slow, rather
than spead2's packet handling.

The most likely candidate is the calculation of the power. Let's write a more
optimal version of that. In the Python case, we use numba_ to compile code on
the fly, and in both cases, we're doing the accumulation in integers to avoid
the cost of integer-to-float conversions for every element. We'll also print
out the number of heaps received, just to ensure we didn't miss any. The
warnings about incomplete heaps should tell us, but if we lose 100% of the
packets in a heap we won't get any warning about it from spead2.

.. tab-set-code::

 .. code-block:: python

    import numba
    ...
    @numba.njit
    def mean_power(adc_samples):
        total = np.int64(0)
        for i in range(len(adc_samples)):
            sample = np.int64(adc_samples[i])
            total += sample * sample
        return np.float64(total) / len(adc_samples)

    def main():
        ...
        n_heaps = 0
        # Run it once to trigger compilation for int8
        mean_power(np.ones(1, np.int8))  # Trigger JIT
        for heap in stream:
            ...
            power = mean_power(item_group["adc_samples"].value)
            n_heaps += 1
            print(f"Timestamp: {timestamp:<10} Power: {power:.2f}")
        print(f"Received {n_heaps} heaps")

 .. code-block:: c++

    #if defined(__GNUC__) && defined(__x86_64__)
    // Compile this function with AVX2 for better performance. Remove this if your
    // CPU does not support AVX2 (e.g., if you get an Illegal Instruction error).
    [[gnu::target("avx2")]]
    #endif
    static double mean_power(const std::int8_t *adc_samples, std::size_t length)
    {
        std::int64_t sum = 0;
        for (std::size_t i = 0; i < length; i++)
            sum += adc_samples[i] * adc_samples[i];
        return double(sum) / length;
    }

    int main()
    {
        ...
        std::int64_t n_heaps = 0;
        for (const spead2::recv::heap &heap : stream)
        {
            ...
            if (timestamp >= 0 && adc_samples != nullptr)
            {
                double power = mean_power(adc_samples, length);
                n_heaps++;
                ...
            }
        }
        std::cout << "Received " << n_heaps << " heaps\n";
        return 0;
    }

On my machine, the receiver now keeps up with the sender and receives all
1000 heaps, although it is somewhat tight so you might get different
results.

.. _numba: http://numba.org/

Full code
---------
.. tab-set-code::

   .. literalinclude:: ../../examples/tutorial/tut_7_recv_power.py
      :language: python

   .. literalinclude:: ../../examples/tutorial/tut_7_recv_power.cpp
      :language: c++
