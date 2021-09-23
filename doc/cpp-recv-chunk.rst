Chunking receiver
=================

.. warning::

   This feature is **experimental**. Future releases of spead2 may change it
   in backwards-incompatible ways, and it could even be removed entirely.

For an overview, refer to :doc:`recv-chunk`. This page is a reference for the
C++ API.

.. doxygenstruct:: spead2::recv::chunk_place_data
   :members:

.. doxygentypedef:: spead2::recv::chunk_place_function

.. cpp:type:: std::function<std::unique_ptr<chunk>(std::int64_t chunk_id, std::uint64_t *batch_stats)> chunk_allocate_function

   Callback to obtain storage for a new chunk.

.. doxygentypedef:: spead2::recv::chunk_ready_function

.. doxygenclass:: spead2::recv::chunk
   :members:

.. doxygenclass:: spead2::recv::chunk_stream_config
   :members:

.. doxygenclass:: spead2::recv::chunk_stream
   :members: chunk_stream, get_chunk_config, get_heap_metadata

Ringbuffer convenience API
--------------------------

.. doxygenclass:: spead2::recv::chunk_ring_stream
   :members: chunk_ring_stream, add_free_chunk, get_data_ringbuffer, get_free_ringbuffer
