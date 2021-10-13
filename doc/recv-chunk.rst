Chunking receiver
=================

.. warning::

   This feature is **experimental**. Future releases of spead2 may change it
   in backwards-incompatible ways, and it could even be removed entirely.

For some high-bandwidth use cases, dealing with heaps one at a time is not
practical. For example, transferring data to an accelerator (such as a GPU) or
another process may have a high overhead that needs to be amortised over many
heaps. This is particularly true when using the Python API, as transferring a
heap across the C++/Python boundary has a non-trivial overhead.

Furthermore, when grouping heaps together, it may be useful to assemble them
according to their semantic position, rather than just in the order they
arrived. For example, if the heaps represent samples in time, it is useful to
place them within larger buffers according to their timestamps. The
:dfn:`chunking receiver` classes support these use cases.

The main limitation of these classes is that only the heap payload is
accessible in the output. The heap items are used only to determine where each
heap belongs and are then discarded, and descriptors are not accessible. It is
also necessary for the receiver to know in advance how big the heaps will be.
It is thus best suited to application-specific receivers that have prior
knowledge of the heap layout.

Heaps are organised into :dfn:`chunks`, which collect the payload from
multiple heaps into a contiguous region of memory. There are assumed to be a
continuous sequence of chunks, with consecutive 64-bit IDs. The user provides
a callback (the :dfn:`place callback`) which determines for each heap

- the chunk ID;
- the heap :dfn:`index` within the chunk;
- the :dfn:`offset` within the chunk.

The heap index is used to report which heaps are received, via a boolean array.
The indices for a chunk should thus be consecutive integers starting from 0
(gaps are allowed, but will waste memory in the array). The offset is the
byte offset within the storage for the chunk. The callback may also indicate
that the heap should be ignored by returning a chunk ID of -1 (which is the
value on entry, so this is the effect if the callback does not change the
chunk ID).

Chunks are assumed to be received approximately in order (increasing ID)
without gaps, but with a tolerance specified as a maximum number of contiguous
chunks to have under construction at one time. If a heap is received whose
chunk ID is too low, it is dropped.

When the first packet of the first heap of a chunk is seen, a user-provided
callback (the :dfn:`allocation callback`) is used to obtain the storage for
the chunk. It is responsible for obtaining the memory for both the payload and
the boolean array indicating the present heaps. It must also zero out the
boolean array, and it may choose to fill the payload as well. Note that when
an incomplete heap is received, it will not be marked present in the array,
but some of the payload bytes may still have been written.

Chunks may also be requested from the callback even when no packets have been
seen for them. This occurs when the new chunk ID is not contiguous with the
previously seen chunk ID. Chunks are back-filled (up to the window size) so
that they are present should an older heap arrive later. This can also happen
for the very first packet of the stream, but it is limited to chunks with
non-negative IDs. Thus, if the first packet corresponds to chunk 0, there will
not be any back-filling. It is worth noting that the chunk IDs used in the
callback are strictly monotonic.

If the callback returns a null pointer, all work on this chunk is silently
skipped. This is intended only for use in shutdown code (i.e., during the call
to ``stop``) to avoid needing to create chunks that will never be consumed.

Once a chunk is aged out (by the arrival of newer chunks), or when the stream
is stopped, it is passed to another callback (the :dfn:`ready callback`) for
processing.

.. _packet-presence:

Packet presence
---------------
Instead of only getting information on which heaps were successfully received,
it is possible to instead get information about which *packets* were received,
even if some packets from a heap are missing. This is only possible if the amount
of payload in each packet is known in advance. The payload offset item is
divided by the expected payload size and added to the heap offset returned by
the callback before being used.

When using this feature one may wish to enable the ``allow_out_of_order`` flag
when configuring the stream, so that the loss of a packet in the middle of a
heap does not prevent the following packets from being processed.

Ringbuffer convenience API
--------------------------
A subclass is provided that takes care of the allocation and ready callbacks
using ringbuffers of chunks (for Python, this is the only API provided). This
is aimed at use with a fixed pool of chunks that is recycled. Two ringbuffers
are used: one moves completed chunks from the stream to the consumer, and the
other returns chunks that are no longer needed to the stream. It is
strongly recommended that both ringbuffers have capacity that is equal to the
maximum number of chunks in the system, so they they never fill up and
block (each ringbuffer slot only requires space for a single pointer, so the
cost is low).

While it is possible to add freed chunks directly to the free ringbuffer, a
:cpp:func:`spead2::recv::chunk_ring_stream::add_free_chunk` convenience function
takes care of some details. It zeros out the heap presence flags, and if the
ringbuffer has been stopped, it fails silently rather than throwing an
exception. This avoids the need for exception-handling code when the stream is
being shut down.

The ringbuffers are passed to the stream constructor, and can be shared
between streams. This provides a mechanism to have a shared pool of free
chunks, or to multiplex chunks from several streams together to a single
consumer. In the latter case, it is often necessary to know which stream
produced the chunk. Set the :cpp:func:`stream ID
<spead2::recv::stream_config::set_stream_id>` when constructing each stream;
it is available as an attribute of the corresponding chunks.

When the stream is stopped by the user, both ringbuffers are stopped too. This
makes sharing ringbuffers appropriate only when the streams have the same
lifetime. However (since version 3.6.0), if a stream is stopped due to network
activity, the free ringbuffer is not stopped, and the data ringbuffer is only
stopped if this was the last stream sharing the ringbuffer.

Examples
--------
The spead2 source distribution includes a number of examples that use this
API, in both C++ and Python.

Advice for senders
------------------
The ready callback uses items in the first received packet of each heap. It's
thus critical that the first packet (and ideally, every packet) of the heap
contains immediate items necessary for correctly placing the heap. Senders can
ensure this by using :attr:`spead2.send.Heap.repeat_pointers`.

Item descriptors form part of the heap payload, and hence would get mixed up
with the actual data in the payload. It is thus best to separate heaps into
those that only have descriptors and those that only have data. One could also
eliminate descriptors entirely, but they are quite useful for debugging. If
descriptors are used, receivers must be prepared to ignore those heaps.
