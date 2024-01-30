#!/usr/bin/env python3

# Copyright 2023-2024 National Research Foundation (SARAO)
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

import argparse
import ctypes

import numba
import numpy as np
import scipy
from numba import types

import spead2.recv
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data


@numba.njit
def mean_power(adc_samples, present):
    total = np.int64(0)
    n = 0
    for i in range(len(present)):
        if present[i]:
            for j in range(adc_samples.shape[1]):
                sample = np.int64(adc_samples[i, j])
                total += sample * sample
            n += adc_samples.shape[1]
    return np.float64(total) / n


@numba.cfunc(
    types.void(types.CPointer(chunk_place_data), types.size_t, types.CPointer(types.int64)),
    nopython=True,
)
def place_callback(data_ptr, data_size, sizes_ptr):
    data = numba.carray(data_ptr, 1)
    items = numba.carray(intp_to_voidptr(data[0].items), 2, dtype=np.int64)
    sizes = numba.carray(sizes_ptr, 2)
    payload_size = items[0]
    timestamp = items[1]
    heap_size = sizes[0]
    chunk_size = sizes[1]
    if timestamp >= 0 and payload_size == heap_size:
        data[0].chunk_id = timestamp // chunk_size
        data[0].heap_offset = timestamp % chunk_size
        data[0].heap_index = data[0].heap_offset // heap_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--heap-size", type=int, default=1024 * 1024)
    parser.add_argument("port", type=int)
    args = parser.parse_args()

    heap_size = args.heap_size
    chunk_size = 1024 * 1024  # Preliminary value
    chunk_heaps = max(1, chunk_size // heap_size)
    chunk_size = chunk_heaps * heap_size  # Final value

    thread_pool = spead2.ThreadPool(1, [2])
    spead2.ThreadPool.set_affinity(3)
    config = spead2.recv.StreamConfig(max_heaps=2)
    user_data = np.array([heap_size, chunk_size], np.int64)
    chunk_config = spead2.recv.ChunkStreamConfig(
        items=[spead2.HEAP_LENGTH_ID, 0x1600],
        max_chunks=1,
        place=scipy.LowLevelCallable(
            place_callback.ctypes,
            user_data.ctypes.data_as(ctypes.c_void_p),
            "void (void *, size_t, void *)",
        ),
    )
    data_ring = spead2.recv.ChunkRingbuffer(2)
    free_ring = spead2.recv.ChunkRingbuffer(4)
    stream = spead2.recv.ChunkRingStream(thread_pool, config, chunk_config, data_ring, free_ring)
    for _ in range(free_ring.maxsize):
        chunk = spead2.recv.Chunk(
            data=np.zeros((chunk_heaps, heap_size), np.int8),
            present=np.zeros(chunk_heaps, np.uint8),
        )
        stream.add_free_chunk(chunk)

    stream.add_udp_reader(args.port)
    n_heaps = 0
    # Run it once to trigger compilation for int8
    mean_power(np.ones((1, 1), np.int8), np.ones(1, np.uint8))
    for chunk in data_ring:
        timestamp = chunk.chunk_id * chunk_size
        n = int(np.sum(chunk.present, dtype=np.int64))
        if n > 0:
            power = mean_power(chunk.data, chunk.present)
            n_heaps += n
            print(f"Timestamp: {timestamp:<10} Power: {power:.2f}")
        stream.add_free_chunk(chunk)
    print(f"Received {n_heaps} heaps")


if __name__ == "__main__":
    main()
