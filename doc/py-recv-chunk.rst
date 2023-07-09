Chunking receiver
=================

For an overview, refer to :doc:`recv-chunk`. This page is a reference for the
Python API.

.. _place-callback:

Writing a place callback
------------------------
A callback is needed to determine which chunk each heap belongs to and where
it fits into that chunk. The callback is made from a C++ thread, so it cannot
be written in pure Python, or even use the Python C API. It needs to be
compiled code. The callback code should also not attempt to acquire the Global
Interpreter Lock (GIL), as it may lead to a deadlock.

Once you've written the function (see below), it needs to be passed to spead2.
The easiest way to do this is with :py:class:`scipy.LowLevelCallable`.
However, it's not strictly necessary to use scipy. The callback must be
represented with a tuple whose first element is a :c:type:`PyCapsule`.
The other elements are not used, but a reference to the tuple is held so this
can be used to keep things alive). The capsule's pointer must be a function
pointer to the code, the name must be the function signature, and a
user-defined pointer may be set as the capsule's context.

For the place callback, the signature must be one of the following (exactly,
with no whitespace changes):

- ``"void (void *, size_t)"``
- ``"void (void *, size_t, void *)"``

In the latter case, the capsule's context is provided as the final argument.
The first two arguments are a pointer to
:cpp:struct:`spead2::recv::chunk_place_data` and the size of that structure.

There are lots of ways to write compiled code and access the functions from
Python: ctypes, cffi, cython, pybind11 are some of the options. One for which
spead2 provided some extra support is numba:

.. autodata:: spead2.recv.numba.chunk_place_data
   :no-value:

.. autofunction:: spead2.numba.intp_to_voidptr

Reference
---------

.. py:class:: spead2.recv.Chunk(**kwargs)

   The attributes can also be used as keywords arguments to the constructor.
   This class is designed to allow subclassing, and subclass properties will
   round-trip through the stream.

   .. py:attribute:: data

   Data storage for a chunk. This can be set to any object that supports the
   Python buffer protocol, as long as it is contiguous and writable. Examples
   include (contiguous) numpy arrays, :py:class:`bytearray` and
   :py:class:`memoryview`. It can also be set to ``None`` to clear it.

   .. py:attribute:: present

   Data storage for flags indicating presence of heaps within the chunk. This
   can be set to any object that supports the Python buffer protocol, as long
   as it is contiguous and writable. It can also be set to ``None`` to clear
   it.

   .. py:attribute:: extra

   Data storage for extra data to be written by the place callback. This
   can be set to any object that supports the Python buffer protocol, as long
   as it is contiguous and writable. It can also be set to ``None`` to clear
   it.

   .. py:attribute:: chunk_id

   The chunk ID determined by the placement function.

   .. py:attribute:: stream_id

   Stream ID of the stream from which the chunk originated.

.. py:class:: spead2.recv.ChunkStreamConfig(**kwargs)

   Parameters for a :py:class:`~spead2.recv.ChunkStream`. The configuration options
   can either be passed to the constructor (as keyword arguments) or set as
   properties after construction.

   :param List[int] items:
     The items whose immediate values should be passed to the place function.
     Accessing this property returns a copy, so it cannot be updated with
     ``append`` or other mutating operations. Assign a complete list.
   :param int max_chunks:
     The maximum number of chunks that can be live at the same time. A value of
     1 means that heaps must be received in order: once a chunk is started, no
     heaps from a previous chunk will be accepted.
   :param tuple place:
     See :ref:`place-callback`.
   :param int max_heap_extra:
     The maximum amount of data a placement function may write to
     :cpp:member:`spead2::recv::chunk_place_data::extra`.
   :raises ValueError: if `max_chunks` is zero.

   .. py:method:: enable_packet_presence(payload_size: int)

   Enable the :ref:`packet presence feature <packet-presence>`.
   The payload offset of each packet is divided by `payload_size` and added
   to the heap index before indexing :py:attr:`spead2.recv.Chunk.present`.

   .. py:method:: disable_packet_presence()

   Disable the packet presence feature enabled by :py:meth:`enable_packet_presence`.

   .. py:attribute:: packet_presence_payload_size

   The `payload_size` if packet presence is enabled, or 0 if not.

   .. py:data:: DEFAULT_MAX_CHUNKS

   Default value for :py:attr:`max_chunks`.

.. py:class:: spead2.recv.ChunkRingbuffer(maxsize)

   Ringbuffer holding :py:class:`.Chunk`\ s. The interface is modelled on
   :class:`Queue`, although the exceptions are different. It also implements
   the iterator protocol.

   Once a chunk has been added to a ringbuffer it should not be accessed again
   until it is retrieved from a ringbuffer (either the same one, or more
   typically, a different one after it has been filled in by
   :class:`.ChunkRingStream`).

   .. py:attribute:: maxsize

   Maximum capacity of the ringbuffer.

   .. py:attribute:: data_fd

   A file descriptor that is readable when there is data available. This will
   not normally be used directly, but is used in the implementation of
   :class:`spead2.recv.asyncio.ChunkRingbuffer`.

   .. py:attribute:: space_fd

   A file descriptor that is readable when there is free space available. This
   will not normally be used directly, but is used in the implementation of
   :class:`spead2.recv.asyncio.ChunkRingbuffer`.

   .. py:method:: qsize()

   The current number of items in the ringbuffer.

   .. py:method:: empty()

   True if the ringbuffer is empty, otherwise false.

   .. py:method:: full()

   True if the ringbuffer is full, otherwise false.

   .. py:method:: get()

   Retrieve an item from the ringbuffer, blocking if necessary.

   :raises spead2.Stopped: if the ringbuffer was stopped before an item became available.

   .. py:method:: get_nowait()

   Retrieve an item from the ringbuffer, raising an exception if none is available.

   :raises spead2.Stopped: if the ringbuffer is stopped and empty.
   :raises spead2.Empty: if the ringbuffer is empty.

   .. py:method:: put(chunk)

   Put an item into the ringbuffer, blocking until there is space if necessary.

   :raises spead2.Stopped: if the ringbuffer was stopped before space became available.

   .. py:method:: put_nowait(chunk)

   Put an item into the ringbuffer, raising an exception if there is no space available.

   :raises spead2.Stopped: if the ringbuffer is stopped.
   :raises spead2.Full: if the ringbuffer is full.

   .. py:method:: stop()

   Shut down the ringbuffer. Producers will no longer be able to add new items.
   Consumers will be able to retrieve existing items, after which they will
   receive :exc:`spead2.Stopped`, and iterators will terminate.

   Returns true if this call stopped the ringbuffer, otherwise false.

   .. py:method:: add_producer()

   Register a new producer. Producers only need to call this if they want to
   call :meth:`remove_producer`.

   .. py:method:: remove_producer()

   Indicate that a producer registered with :meth:`add_producer` is finished
   with the ringbuffer. If this was the last producer, the ringbuffer is
   stopped. Returns true if this call stopped the ringbuffer, otherwise false.

.. autoclass:: spead2.recv.asyncio.ChunkRingbuffer
   :members:

.. py:class:: spead2.recv.ChunkRingStream(thread_pool, config, chunk_config, data_ringbuffer, free_ringbuffer)

   Stream that works on chunks. While it is not a direct subclass, it
   implements most of the same functions as :py:class:`spead2.recv.Stream`,
   in particular for adding transports.

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param config: Stream configuration
   :type config: :py:class:`spead2.recv.StreamConfig`
   :param chunk_config: Chunking configuration
   :type chunk_config: :py:class:`spead2.recv.ChunkStreamConfig`
   :param data_ringbuffer: Ringbuffer onto which the stream will place completed chunks.
   :type data_ringbuffer: :py:class:`spead2.recv.ChunkRingbuffer`
   :param free_ringbuffer: Ringbuffer from which the stream will obtain new chunks.
   :type free_ringbuffer: :py:class:`spead2.recv.ChunkRingbuffer`

   .. py:attribute:: data_ringbuffer

      The data ringbuffer given to the constructor.

   .. py:attribute:: free_ringbuffer

      The free ringbuffer given to the constructor.

   .. py:method:: add_free_chunk(chunk)

      Add a chunk to the free ringbuffer. This takes care of zeroing out the
      :py:attr:`.Chunk.present` array, and it will suppress the
      :exc:`spead2.Stopped` exception if the free ringbuffer has been stopped.

      If the free ring is full, it will raise :exc:`spead2.Full` rather than
      blocking. The free ringbuffer should be constructed with enough slots that
      this does not happen.
