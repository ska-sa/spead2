Changelog
=========

.. rubric:: Version 0.2.0

- **backwards-incompatible change**: for sending, the heap count is now tracked
  internally by the stream, rather than an attribute of the heap. This affects
  both C++ and Python bindings, although Python code that always uses
  :py:class:`~spead2.send.HeapGenerator` rather than directly creating heaps
  will not be affected.

- The :py:class:`~spead2.send.HeapGenerator` is extended to allow items to be
  added to an existing heap and to give finer control over whether descriptors
  and/or values are put in the heap.

- Fixes a bug that caused some values to be cast to non-native endian.

- Added overloaded equality tests on Flavour objects.

.. rubric:: Version 0.1.2

- Coerce values to int for legacy 'u' and 'i' fields

- Fix flavour selection in example code

.. rubric:: Version 0.1.1

- Fixes to support OS X

.. rubric:: Version 0.1.0

- First public release
