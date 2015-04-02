Performance tips
================

Heap lifetime
-------------
All the payload for a heap is stored in a single memory allocation, and where
possible, items reference this memory. This means that the entire heap remains
live as long as any of the values encoded in it are live. Thus, a small but
seldom-changing value can cause a very large heap to remain live long after
the rest of the values in that heap have been replaced. On the sending side
this can be avoided by only grouping items into a heap if they are updated at
the same frequency. If it is not possible to change the sender, the receiver
can copy values.

Alignment
---------
Because items directly reference the received data (where possible), it is
possible that data will be misaligned. While numpy allows this, it could make
access to the data inefficient. Ideally the sender will ensure that all values
are aligned within the heap payload, but unfortunately the bindings do not
currently provide a way to ensure this. If only a single addressed item is
placed in a heap, it will be naturally aligned.

Endianness
----------
When using numpy builtin types, data are converted to native endian when they
are received, to allow for more efficient operations on them. This can
significantly reduce the maximum rate at which packets are received. Thus,
using the native endian on the wire will give better performance.

Data format
-----------
Using the `dtype` parameter to the :py:class:`spead2.Item` constructor is
highly recommended. While the `format` parameter is more generic, it uses a
much slower path for encoding and decoding. In some cases it can determine an
equivalent `dtype` and use the fast path, but relying on this is not
recommended. The `dtype` approach is also the only way to transmit in
little-endian, which will be faster when the host is little-endian (such as
x86).
