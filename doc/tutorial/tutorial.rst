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

The tutorial is best followed on GNU/Linux on an x86-64 system. The code should
be portable to other POSIX systems (such as MacOS), but that's untested, and
some of the behaviour that's pointed out (particularly performance) will be
different.

The code that is developed in each tutorial appears in full at the end of the
tutorial, and can also be found in the `repository`_.

.. _repository: https://github.com/ska-sa/spead2/tree/master/examples/tutorial

.. toctree::
   :maxdepth: 1
   :numbered: 1

   tut-1-spead-intro
   tut-2-send
   tut-3-recv
   tut-4-send-perf
   tut-5-send-pipeline
   tut-6-send-pktsize
   tut-7-recv-power
   tut-8-send-reuse-memory
   tut-9-recv-memory-pool
   tut-10-send-reuse-heaps
   tut-11-send-batch-heaps
   tut-12-recv-chunks
