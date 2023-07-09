Chunking stream groups
======================

For an overview, refer to :doc:`recv-chunk-group`. This page is a reference for the
Python API. It extends the API for :doc:`chunks <py-recv-chunk>`.

.. py:class:: spead2.recv.ChunkStreamGroupConfig(**kwargs)

   Parameters for a chunk stream group. The configuration options
   can either be passed to the constructor (as keyword arguments) or set as
   properties after construction.

   :param int max_chunks:
     The maximum number of chunks that can be live at the same time.
   :param EvictionMode eviction_mode:
     The chunk eviction mode.

   .. py:class:: EvictionMode

     Eviction mode when it is necessary to advance the group window. See
     the :doc:`overview <recv-chunk-group>` for more details.

     .. py:attribute:: LOSSY

        force streams to release incomplete chunks

     .. py:attribute:: LOSSLESS

        a chunk will only be marked ready when all streams have marked it
        ready

.. py:class:: spead2.recv.ChunkStreamRingGroup(config, data_ringbuffer, free_ringbuffer)

   Stream group that uses ringbuffers to manage chunks.

   When a fresh chunk is needed, it is retrieved from a ringbuffer of free
   chunks (the "free ring"). When a chunk is flushed, it is pushed to a "data
   ring". These may be shared between groups, but both will be stopped as soon
   as any of the members streams are stopped. The intended use case is
   parallel groups that are started and stopped together.

   It behaves like a :py:class:`~collections.abc.Sequence` of the contained
   streams.

   :param config: Group configuration
   :type config: :py:class:`spead2.recv.ChunkStreamGroupConfig`
   :param data_ringbuffer: Ringbuffer onto which the completed chunks are placed.
   :type data_ringbuffer: :py:class:`spead2.recv.ChunkRingbuffer`
   :param free_ringbuffer: Ringbuffer from which new chunks are obtained.
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

   .. py:method:: emplace_back(thread_pool, config, chunk_stream_config)

      Add a new stream.

      :param thread_pool: Thread pool handling the I/O
      :type thread_pool: :py:class:`spead2.ThreadPool`
      :param config: Stream configuration
      :type config: :py:class:`spead2.recv.StreamConfig`
      :param chunk_config: Chunking configuration
      :type chunk_config: :py:class:`spead2.recv.ChunkStreamConfig`
      :rtype: :py:class:`spead2.recv.ChunkStreamGroupMember`

.. py:class:: spead2.recv.ChunkStreamGroupMember

   A component stream in a :py:class:`~spead2.recv.ChunkStreamRingGroup`.
   This class cannot be instantiated directly. Use
   :py:meth:`.ChunkStreamRingGroup.emplace_back` instead.

   It provides the same methods for adding readers as
   :py:class:`spead2.recv.Stream`.
