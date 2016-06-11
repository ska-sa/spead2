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

   .. py:function:: is_start_of_stream()

      Returns true if the packet contains a stream start control item.

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

.. py:class:: spead2.recv.Stream(thread_pool, bug_compat=0, max_heaps=4, ring_heaps=4)

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param int bug_compat: Bug compatibility flags (see :ref:`py-flavour`)
   :param int max_heaps: The number of partial heaps that can be live at one
     time. This affects how intermingled heaps can be (due to out-of-order
     packet delivery) before heaps get dropped.
   :param int ring_heaps: The capacity of the ring buffer between the network
     threads and the consumer. Increasing this may reduce lock contention at
     the cost of more memory usage.

   .. py:method:: set_memory_allocator(allocator)

      Set or change the memory allocator for a stream. See
      :ref:`py-memory-allocators` for details.

      :param pool: New memory allocator
      :type pool: :py:class:`spead2.MemoryAllocator`

   .. py:method:: set_memcpy(id)

      Set the method used to copy data from the network to the heap. The
      default is :py:const:`MEMCPY_STD`. This can be changed to
      :py:const:`MEMCPY_NONTEMPORAL`, which writes to the destination with a
      non-temporal cache hint (if SSE2 is enabled at compile time). This can
      improve performance with large heaps if the data is not going to be used
      immediately, by reducing cache pollution. Be careful when benchmarking:
      receiving heaps will generally appear faster, but it can slow down
      subsequent processing of the heap because it will not be cached.

      :param id: Identifier for the copy function
      :type id: {:py:const:`MEMCPY_STD`, :py:const:`MEMCPY_NONTEMPORAL`}

   .. py:method:: add_buffer_reader(buffer)

      Feed data from an object implementing the buffer protocol.

   .. py:method:: add_udp_reader(port, max_size=DEFAULT_UDP_MAX_SIZE, buffer_size=DEFAULT_UDP_BUFFER_SIZE, bind_hostname='', socket=None)

      Feed data from a UDP port.

      :param int port: UDP port number
      :param int max_size: Largest packet size that will be accepted.
      :param int buffer_size: Kernel socket buffer size. If this is 0, the OS
        default is used. If a buffer this large cannot be allocated, a warning
        will be logged, but there will not be an error.
      :param str bind_hostname: If specified, the socket will be bound to the
        first IP address found by resolving the given hostname. If this is a
        multicast group, then it will also subscribe to this multicast group.
      :param socket.socket socket: If specified, this socket is used rather
        than a new one. The socket must be open but unbound. The caller must
        not use this socket any further, although it is not necessary to keep
        it alive. This is mainly useful for fine-tuning socket options such
        as multicast subscriptions.

   .. py:method:: add_udp_reader(multicast_group, port, max_size=DEFAULT_UDP_MAX_SIZE, buffer_size=DEFAULT_UDP_BUFFER_SIZE, interface_address)

      Feed data from a UDP port with multicast (IPv4 only).

      :param str multicast_group: Hostname/IP address of the multicast group to subscribe to
      :param int port: UDP port number
      :param int max_size: Largest packet size that will be accepted.
      :param int buffer_size: Kernel socket buffer size. If this is 0, the OS
        default is used. If a buffer this large cannot be allocated, a warning
        will be logged, but there will not be an error.
      :param str interface_address: Hostname/IP address of the interface which
        will be subscribed, or the empty string to let the OS decide.

   .. py:method:: add_udp_reader(multicast_group, port, max_size=DEFAULT_UDP_MAX_SIZE, buffer_size=DEFAULT_UDP_BUFFER_SIZE, interface_index)

      Feed data from a UDP port with multicast (IPv6 only).

      :param str multicast_group: Hostname/IP address of the multicast group to subscribe to
      :param int port: UDP port number
      :param int max_size: Largest packet size that will be accepted.
      :param int buffer_size: Kernel socket buffer size. If this is 0, the OS
        default is used. If a buffer this large cannot be allocated, a warning
        will be logged, but there will not be an error.
      :param str interface_index: Index of the interface which will be
        subscribed, or 0 to let the OS decide.

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

.. _trollius: http://trollius.readthedocs.io/
.. _twisted: https://twistedmatrix.com/trac/

.. _py-memory-allocators:

Memory allocators
^^^^^^^^^^^^^^^^^
To allow for performance tuning, it is possible to use an alternative memory
allocator for heap payloads. A few allocator classes are provided; new classes
must currently be written in C++. The default (which is also the base class
for all allocators) is :py:class:`spead2.MemoryAllocator`, which has no
constructor arguments or methods. An alternative is
:py:class:`spead2.MmapAllocator`.

.. py:class:: spead2.MmapAllocator(flags=0)

    An allocator using :manpage:`mmap(2)`. This may be slightly faster for large
    allocations, and allows setting custom mmap flags. This is mainly intended
    for use with the C++ API, but is exposed to Python as well.

    :param int flags:
        Extra flags to pass to :manpage:`mmap(2)`. Finding the numeric values
        for OS-specific flags is left as a problem for the user.

The most important custom allocator is :py:class:`spead2.MemoryPool`. It allocates
from a pool, rather than directly from the system. This can lead to
significant performance improvements when the allocations are large enough
that the C library allocator does not recycle the memory itself, but instead
requests memory from the kernel.

A memory pool has a range of sizes that it will handle from its pool, by
allocating the upper bound size. Thus, setting too wide a range will waste
memory, while setting too narrow a range will prevent the memory pool from
being used at all. A memory pool is best suited for cases where the heaps are
all roughly the same size.

A memory pool can optionally use a background task (scheduled onto a thread
pool) to replenish the pool when it gets low. This is useful when heaps are
being captured and stored indefinitely rather than processed and released.

.. py:class:: spead2.MemoryPool(thread_pool, lower, upper, max_free, initial, low_water, allocator=None)

   Constructor. One can omit `thread_pool` and `low_water` to skip the
   background refilling.

   :param ThreadPool thread_pool: thread pool used for
     refilling the memory pool
   :param int lower: Minimum allocation size to handle with the pool
   :param int upper: Size of allocations to make
   :param int max_free: Maximum number of allocations held in the pool
   :param int initial: Number of allocations to put in the free pool
     initially.
   :param int low_water: When fewer than this many buffers remain, the
     background task will be started and allocate new memory until `initial`
     buffers are available.
   :param MemoryAllocator allocator: Underlying memory allocator
