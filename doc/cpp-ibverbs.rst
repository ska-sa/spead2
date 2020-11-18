Support for ibverbs
===================
The support for libibverbs is essentially the same as for :doc:`Python
<py-ibverbs>`, with the same limitations. The programmatic interface is via
the :cpp:class:`spead2::recv::udp_ibv_reader` and
:cpp:class:`spead2::send::udp_ibv_stream` classes:

.. doxygenclass:: spead2::recv::udp_ibv_config
   :members:

.. doxygenclass:: spead2::recv::udp_ibv_reader
   :members: udp_ibv_reader

.. doxygenclass:: spead2::send::udp_ibv_config
   :members:

.. doxygenclass:: spead2::send::udp_ibv_stream
   :members: udp_ibv_stream

PeerDirect
----------
The pointer given to
:cpp:function:`spead2::send::udp_ibv_config::add_memory_region` is passed to
:man:`ibv_reg_mr(2)`. When using a Mellanox NIC, this can be a pointer that is
handled by PeerDirect, such as a GPU device pointer. This can be used to
transfer data directly from a GPU to the network without passing though the
CPU.

This approach does need some care, because the spead2 implementation will fall
back to copying if a packet contains too many discontiguous pieces of memory.
It will be safe as long as there is only one item in a heap that uses a
registered memory region, or as long as all such items are at least as big as
the packet size.

For an example of this, see :file:`examples/gpudirect_example.cu` in the spead2
source distribution.
