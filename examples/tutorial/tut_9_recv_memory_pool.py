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

import numba
import numpy as np

import spead2.recv


@numba.njit
def mean_power(adc_samples):
    total = np.int64(0)
    for i in range(len(adc_samples)):
        sample = np.int64(adc_samples[i])
        total += sample * sample
    return np.float64(total) / len(adc_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--heap-size", type=int, default=1024 * 1024)
    parser.add_argument("port", type=int)
    args = parser.parse_args()

    thread_pool = spead2.ThreadPool()
    config = spead2.recv.StreamConfig(max_heaps=2)
    ring_config = spead2.recv.RingStreamConfig(heaps=2)
    pool_heaps = config.max_heaps + ring_config.heaps + 1
    config.memory_allocator = spead2.MemoryPool(
        lower=0,
        upper=args.heap_size + 8192,
        max_free=pool_heaps,
        initial=pool_heaps,
    )
    stream = spead2.recv.Stream(thread_pool, config, ring_config)
    stream.add_udp_reader(args.port)
    item_group = spead2.ItemGroup()
    n_heaps = 0
    # Run it once to trigger compilation for int8
    mean_power(np.ones(1, np.int8))
    for heap in stream:
        item_group.update(heap)
        timestamp = item_group["timestamp"].value
        power = mean_power(item_group["adc_samples"].value)
        n_heaps += 1
        print(f"Timestamp: {timestamp:<10} Power: {power:.2f}")
    print(f"Received {n_heaps} heaps")


if __name__ == "__main__":
    main()
