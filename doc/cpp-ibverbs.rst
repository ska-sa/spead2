Support for ibverbs
===================
The support for libibverbs is essentially the same as for :doc:`Python
<py-ibverbs>`, with the same limitations. The programmatic interface is via
the :cpp:class:`spead2::recv::udp_ibv_reader` and
:cpp:class:`spead2::send::udp_ibv_stream` classes:

.. doxygenclass:: spead2::recv::udp_ibv_reader
   :members: udp_ibv_reader

.. doxygenclass:: spead2::send::udp_ibv_stream
   :members: udp_ibv_stream
