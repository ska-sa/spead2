#!/usr/bin/env python3

# Copyright 2021 National Research Foundation (SARAO)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This is an example of using the chunking receive API with ringbuffers. To
# test it, run
# spead2_send localhost:8888 --heaps 1000 --heap-size 65536 --rate 10

from spead2.numba import intp_to_voidptr
import spead2.recv
from spead2.recv.numba import chunk_place_data

import numba
from numba import types
import numpy as np
import scipy

HEAP_PAYLOAD_SIZE = 65536
HEAPS_PER_CHUNK = 64
CHUNK_PAYLOAD_SIZE = HEAPS_PER_CHUNK * HEAP_PAYLOAD_SIZE


@numba.cfunc(types.void(types.CPointer(chunk_place_data), types.uintp), nopython=True)
def chunk_place(data_ptr, data_size):
    data = numba.carray(data_ptr, 1)
    items = numba.carray(intp_to_voidptr(data[0].items), 2, dtype=np.int64)
    heap_cnt = items[0]
    payload_size = items[1]
    # If the payload size doesn't match, discard the heap (could be descriptors etc).
    if payload_size == HEAP_PAYLOAD_SIZE:
        data[0].chunk_id = heap_cnt // HEAPS_PER_CHUNK
        data[0].heap_index = heap_cnt % HEAPS_PER_CHUNK
        data[0].heap_offset = data[0].heap_index * HEAP_PAYLOAD_SIZE


def main():
    MAX_CHUNKS = 4
    place_callback = scipy.LowLevelCallable(
        chunk_place.ctypes,
        signature='void (void *, size_t)'
    )
    chunk_config = spead2.recv.ChunkStreamConfig(
        items=[spead2.HEAP_CNT_ID, spead2.HEAP_LENGTH_ID],
        max_chunks=MAX_CHUNKS,
        place=place_callback)
    data_ring = spead2.recv.ChunkRingbuffer(MAX_CHUNKS)
    free_ring = spead2.recv.ChunkRingbuffer(MAX_CHUNKS)
    stream = spead2.recv.ChunkRingStream(
        spead2.ThreadPool(),
        spead2.recv.StreamConfig(),
        chunk_config,
        data_ring,
        free_ring)
    for i in range(MAX_CHUNKS):
        chunk = spead2.recv.Chunk(
            present=np.empty(HEAPS_PER_CHUNK, np.uint8),
            data=np.empty(CHUNK_PAYLOAD_SIZE, np.uint8)
        )
        stream.add_free_chunk(chunk)
    stream.add_udp_reader(8888, buffer_size=1024 * 1024, bind_hostname='127.0.0.1')
    for chunk in data_ring:
        n_present = np.sum(chunk.present)
        print(
            f"Received chunk {chunk.chunk_id} with "
            f"{n_present} / {HEAPS_PER_CHUNK} heaps")
        stream.add_free_chunk(chunk)


if __name__ == '__main__':
    main()
