Thread pools
------------
The actual sending and receiving of packets is done by separate C threads.
Each stream is associated with a *thread pool*, which is a pool of threads
able to process its packets. In most cases one will use only a single thread
pool for an application, but there may be load-balancing/affinity reasons to
use more. A reasonable number of threads is 1 for a low performance
application, or the smaller of the number of parallel streams and the number
of CPU cores for a high performance application.

.. py:currentmodule:: spead2

.. py:class:: spead2.ThreadPool(threads=1)

   Construct a thread pool and start the threads.

   .. py:method:: stop()

      Shut down the worker threads. Calling this while there are still open
      streams is not advised. In most cases, garbage collection is sufficient.
