.. _py-flavour:

SPEAD flavours
==============
The SPEAD protocol is versioned and within a version allows for multiple
*flavours*, with different numbers of bits for item pointer fields. The spead2
library supports all SPEAD-64-*XX* flavours of version 4, where *XX* is a
multiple of 8.

Furthermore, PySPEAD 0.5.2 has a number of bugs in its implementation of the
protocol, which effectively defines a new protocol. This is treated as part of
the flavour in spead2. Some receive functions have a `bug_compat` parameter
which specifies which of these bugs to maintain compatibility with:

* :py:const:`spead2.BUG_COMPAT_DESCRIPTOR_WIDTHS`: the descriptors are encoded
  with shape and format fields sized as for SPEAD-64-40, regardless of the
  actual flavour.
* :py:const:`spead2.BUG_COMPAT_SHAPE_BIT_1`: the first byte of a shape is set
  to 2 to indicate a variably-sized dimension, instead of 1.
* :py:const:`spead2.BUG_COMPAT_SWAP_ENDIAN`: numpy arrays are encoded/decoded
  in the opposite endianness to that specified in the descriptor.
* :py:const:`spead2.BUG_COMPAT_NO_SCALAR_NUMPY`: scalar items specified with
  a descriptor are transmitted with a descriptor, even if it is possible to
  convert it to a dtype.
* :py:const:`spead2.BUG_COMPAT_PYSPEAD_0_5_2`: all of the above (and any other
  bugs later found in this version of PySPEAD).

For sending, the full flavour is specified by a :py:class:`spead2.Flavour`
object. It allows all the fields to be specified to allow for future
expansion, but :py:class:`ValueError` is raised unless `version` is 4 and
`item_pointer_bits` is 64. There is a default constructor that returns
SPEAD-64-40 with bug compatibility disabled.

.. py:class:: Flavour(version, item_pointer_bits, heap_address_bits, bug_compat=0)

The constructor arguments are available as read-only attributes.
