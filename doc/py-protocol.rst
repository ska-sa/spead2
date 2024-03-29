Mapping of SPEAD protocol to Python
-----------------------------------
* Any descriptor with a numpy header is handled by numpy. The value is
  converted to native endian, but is otherwise left untouched.
* Strings are expected to use ASCII encoding only. At present this is variably
  enforced. Future versions may apply stricter enforcement. This applies to
  names, descriptions, and to values passed with the `c` format code.
* The `c` format code may only be used with length 8, and `f` may only be used
  with lengths 32 or 64.
* The `0` format code is not supported.
* All values sent or received are converted to numpy arrays. If the descriptor
  uses a numpy header, this is the type of the array. Otherwise, a dtype is
  constructed by converting the format code. The following are converted to
  numpy primitive types:

  * u8, u16, u32, u64
  * i8, i16, i32, i64
  * f32, f64
  * b8 (converted to dtype bool)
  * c8 (converted to dtype S1)

  Other fields will be kept as Python objects. If there are multiple fields,
  their names will be generated by numpy (`f0`, `f1`, etc). If all the fields
  convert to native types, a fast path will be used for sending and receiving
  (as fast as using an explicit numpy header).
* At most one element of the shape may indicate a variable-length field,
  whose length will be computed from the size of the item, or zero if any
  other element of the shape is zero.

When transmitting data, a few cases are handled specially:

* If the expected shape is one-dimensional, but the provided value is an
  instance of :py:class:`bytes`, :py:class:`str` or :py:class:`unicode`, it
  will be broken up into its individual characters. This is a convenience for
  sending variable-length strings.
* If the format is a single signed or unsigned integer whose number of bits
  is less than 64 but a multiple of 8, and the value is a zero-dimensional
  numpy array with dtype ``>u8``, the relevant bytes are referenced by the
  heap. The value can later be updated and the same heap sent again without
  creating a new :py:class:`~spead2.send.Heap` object.

When receiving data, some transformations are made:

* A zero-dimensional array is returned as a scalar, rather than a
  zero-dimensional array object.
* If the format is given and is c8 and the array is one-dimensional, it is
  joined together into a Python :py:class:`str`.

Stream control items
--------------------
A heap with the :py:const:`~spead2.CTRL_STREAM_STOP` flag will shut down the
stream, but the heap is not passed on to the application.  Senders should thus
avoid putting any other data in such heaps. These heaps are not automatically
sent; use :py:meth:`spead2.send.HeapGenerator.get_end` to produce such a heap.

In contrast, stream start flags (:py:const:`~spead2.CTRL_STREAM_START`) have no
effect on internal processing. Senders can generate them using
:py:meth:`spead2.send.HeapGenerator.get_start` and receivers can detect them using
:py:meth:`spead2.recv.Heap.is_start_of_stream`.
