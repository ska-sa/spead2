.. _py-bug-compat:

Bug compatibility with PySPEAD
------------------------------
PySPEAD (0.5.2) has a number of bugs in its implementation of the protocol. A
number of the spead2 methods take a `bug_compat` parameter, which is a
bitfield which can be set with the following flags:

* :py:const:`spead2.BUG_COMPAT_DESCRIPTOR_WIDTHS`: the descriptors are encoded
  with shape and format fields sized as for SPEAD-64-40, regardless of the
  actual flavour.
* :py:const:`spead2.BUG_COMPAT_SHAPE_BIT_1`: the first byte of a shape is set
  to 2 to indicate a variably-sized dimension, instead of 1.
* :py:const:`spead2.BUG_COMPAT_SWAP_ENDIAN`: numpy arrays are encoded/decoded
  in the opposite endiannes to that specified in the descriptor.
* :py:const:`spead2.BUG_COMPAT_PYSPEAD_0_5_2`: all of the above (and any other
  bugs later found in this version of PySPEAD).
