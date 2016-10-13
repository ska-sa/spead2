Thread pools
------------
The actual sending and receiving of packets is done by separate C threads.
Each stream is associated with a *thread pool*, which is a pool of threads
able to process its packets. See the :ref:`performance guidelines
<perf-thread-pool>` for advice on how many threads to use.

There is one important consideration for deciding whether streams share a
thread pool: if a received stream is not being consumed, it may block one of
the threads from the thread pool [#]_. Thus, if several streams share a thread
pool, it is important to be responsive to all of them. Deciding that one
stream is temporarily uninteresting and can be discarded while listening only
to another one can thus lead to a deadlock if the two streams share a thread
pool with only one thread.

.. [#] This is a limitation of the current design that will hopefully be
   overcome in future versions.

.. py:currentmodule:: spead2

.. py:class:: spead2.ThreadPool(threads=1, affinity=[])

   Construct a thread pool and start the threads. A list of integers can be
   provided for `affinity` to have the threads bound to specific CPU cores
   (this is only implemented for glibc). If there are fewer values than
   threads, the list is reused cyclically (although in this case you're
   probably better off having fewer threads in this case).

   .. py:method:: stop()

      Shut down the worker threads. Calling this while there are still open
      streams is not advised. In most cases, garbage collection is sufficient.

   .. py:staticmethod:: set_affinity(core)

      Binds the caller to CPU core `core`.
