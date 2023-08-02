#!/usr/bin/env python3

# Copyright 2023 National Research Foundation (SARAO)
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

# This is an example of using the chunk stream group receive API with
# ringbuffers. To test it, run
# spead2_send localhost:8888 localhost:8889 --heaps 1000 --heap-size 65536 --rate 10

import numba
import numpy as np
import scipy
from numba import types

import spead2.recv
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

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
    NUM_STREAMS = 2
    MAX_CHUNKS = 4
    place_callback = scipy.LowLevelCallable(chunk_place.ctypes, signature="void (void *, size_t)")
    chunk_config = spead2.recv.ChunkStreamConfig(
        items=[spead2.HEAP_CNT_ID, spead2.HEAP_LENGTH_ID],
        max_chunks=MAX_CHUNKS,
        place=place_callback,
    )
    group_config = spead2.recv.ChunkStreamGroupConfig(max_chunks=MAX_CHUNKS)
    data_ring = spead2.recv.ChunkRingbuffer(MAX_CHUNKS)
    free_ring = spead2.recv.ChunkRingbuffer(MAX_CHUNKS)
    group = spead2.recv.ChunkStreamRingGroup(group_config, data_ring, free_ring)
    for _ in range(NUM_STREAMS):
        group.emplace_back(spead2.ThreadPool(), spead2.recv.StreamConfig(), chunk_config)
    for _ in range(MAX_CHUNKS):
        chunk = spead2.recv.Chunk(
            present=np.empty(HEAPS_PER_CHUNK, np.uint8), data=np.empty(CHUNK_PAYLOAD_SIZE, np.uint8)
        )
        group.add_free_chunk(chunk)
    for i in range(NUM_STREAMS):
        group[i].add_udp_reader(8888 + i, buffer_size=1024 * 1024, bind_hostname="127.0.0.1")
    for chunk in data_ring:
        n_present = np.sum(chunk.present)
        print(f"Received chunk {chunk.chunk_id} with {n_present} / {HEAPS_PER_CHUNK} heaps")
        group.add_free_chunk(chunk)


if __name__ == "__main__":
    main()
