Receiving
---------
The classes associated with receiving are in the :py:mod:`spead2.recv`
package. A *stream* represents a logical stream, in that packets with
the same heap ID are assumed to belong to the same heap. A stream can have
multiple physical transports.

Streams yield *heaps*, which are the basic units of data transfer and contain
both item descriptors and item values. While it is possible to directly
inspect heaps, this is not recommended or supported. Instead, heaps are
normally passed to :py:meth:`spead2.ItemGroup.update`.

.. py:class:: spead2.recv.Heap

   .. py:attribute:: cnt

      Heap identifier (read-only)

   .. py:attribute:: flavour

      SPEAD flavour used to encode the heap (see :ref:`py-flavour`)

.. note:: Malformed packets (such as an unsupported SPEAD version, or
  inconsistent heap lengths) are dropped, with a log message. However,
  errors in interpreting a fully assembled heap (such as invalid/unsupported
  formats, data of the wrong size and so on) are reported as
  :py:exc:`ValueError` exceptions. Robust code should thus be prepared to
  catch exceptions from heap processing.

Blocking receive
^^^^^^^^^^^^^^^^
To do blocking receive, create a :py:class:`spead2.recv.Stream`, and add
transports to it with :py:meth:`~spead2.recv.Stream.add_buffer_reader` and
:py:meth:`~spead2.recv.Stream.add_udp_reader`. Then either iterate over it,
or repeatedly call :py:meth:`~spead2.recv.Stream.get`.

.. py:class:: spead2.recv.Stream(thread_pool, bug_compat=0, max_heaps=4)

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param int bug_compat: Bug compatibility flags (see :ref:`py-flavour`)
   :param int max_heaps: Size of heap buffers. This determines the number
     of partial heaps that can be live at one time (when a packet from a
     new heap arrives, the old heap is discarded). It also determines the
     number of complete heaps not returned to the user that will be kept
     (new completed heaps will be dropped).

   .. py:method:: set_memory_pool(pool)

      Set or change the memory pool for a stream. See :ref:`py-memory-pools` for
      details.

      :param pool: New memory pool
      :type pool: :py:class:`spead2.MemoryPool`

   .. py:method:: add_buffer_reader(buffer)

      Feed data from an object implementing the buffer protocol.

   .. py:method:: add_udp_reader(port, max_size=9200, buffer_size=8388608, bind_host_name='')

      Feed data from a UDP port.

      :param int port: UDP port number
      :param int max_size: Largest packet size that will be accepted.
      :param int buffer_size: Kernel socket buffer size. If this is 0, the OS
        default is used. If a buffer this large cannot be allocated, a warning
        will be logged, but there will not be an error.
      :param str bind_hostname: If specified, the socket will be bound to the
        first IP address found by resolving the given hostname.

   .. py:method:: get()

      Returns the next heap, blocking if necessary. If the stream has been
      stopped, either by calling :py:meth:`stop` or by receiving a stream
      control packet, it raises :py:exc:`spead2.Stopped`. However, heap that
      were already queued when the stream was stopped are returned first.

      A stream can also be iterated over to yield all heaps.

   .. py:method:: get_nowait()

      Like :py:meth:`get`, but if there is no heap available it raises
      :py:exc:`spead2.Empty`.

   .. py:method:: stop()

      Shut down the stream and close all associated sockets. It is not
      possible to restart a stream once it has been stopped; instead, create a
      new stream.

   .. py:attribute: fd

      The read end of a pipe to which a byte is written when a heap is
      received. **Do not read from this pipe.** It is used for integration
      with asynchronous I/O frameworks (see below).

Asynchronous receive
^^^^^^^^^^^^^^^^^^^^
Asynchronous I/O is supported through trollius_, which is a Python 2 backport
of the Python 3 :py:mod:`asyncio` module. It can be combined with other
asynchronous I/O frameworks like twisted_.

.. py:class:: spead2.recv.trollius.Stream(\*args, \*\*kwargs, loop=None)

   See :py:class:`spead2.recv.Stream` (the base class) for other constructor
   arguments.

   :param loop: Default Trollius event loop for async operations. If not
     specified, uses the default Trollius event loop. Do not call
     `get_nowait` from the base class.

   .. py:method:: get(loop=None)

      Coroutine that yields the next heap, or raises :py:exc:`spead2.Stopped`
      once the stream has been stopped and there is no more data. It is safe
      to have multiple in-flight calls, which will be satisfied in the order
      they were made.

      :param loop: Trollius event loop to use, overriding constructor.

.. _trollius: http://trollius.readthedocs.org/
.. _twisted: https://twistedmatrix.com/trac/

.. _py-memory-pools:

Memory pools
^^^^^^^^^^^^
For high-performance receiving, it is possible to have heaps allocated from a
memory pool rather than directly from the system library. A particular
advantage is that memory can be pre-faulted in advance of the stream arriving,
thus avoiding expensive page faulting when the initial heaps arrive.

A memory pool has a range of sizes that it will handle from its pool, by
allocating the upper bound size. Thus, setting too wide a range will waste
memory, while setting too narrow a range will prevent the memory pool from
being used at all. A memory pool is best suited for cases where the heaps are
all roughly the same size.

.. py:class:: spead2.MemoryPool(lower, upper, max_free, initial)

   :param int lower: Minimum allocation size to handle with the pool
   :param int upper: Size of allocations to make
   :param int max_free: Maximum number of allocations held in the pool
   :param int initial: Number of allocations to put in the free pool
     initially.


