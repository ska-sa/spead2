Tutorial
========
This chapter will start by walking you through creating basic sender and
receiver applications. The resulting application will *not* be
high-performance, and will not cover all the features of the library. From
there we will gradually evolve the applications to improve performance and
demonstrate more features of the library.

Most of the tutorial covers both C++ and Python: just click the appropriate
tab in code examples. Some low-level performance features are not accessible
from the Python API.

The tutorial is best followed on GNU/Linux. The code should be portable to
other POSIX systems (such as MacOS), but that's untested, and some of the
behaviour that's pointed out will be different.

.. toctree::
   :maxdepth: 1
   :numbered:

   tut-spead-intro
   tut-2-send
   tut-3-recv
   tut-4-send-perf
   tut-4-send-pipeline
   tut-5-send-pktsize
   tut-6-recv-power
   tut-7-recv-memory-pool
