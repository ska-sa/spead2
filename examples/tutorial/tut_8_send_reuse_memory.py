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
import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

import spead2.send.asyncio


@dataclass
class State:
    adc_samples: np.ndarray
    future: asyncio.Future[int] = field(default_factory=asyncio.Future)

    def __post_init__(self):
        # Make it safe to wait on the future immediately
        self.future.set_result(0)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--heaps", type=int, default=1000)
    parser.add_argument("-p", "--packet-size", type=int)
    parser.add_argument("-H", "--heap-size", type=int, default=1024 * 1024)
    parser.add_argument("host", type=str)
    parser.add_argument("port", type=int)
    args = parser.parse_args()

    thread_pool = spead2.ThreadPool(1, [0])
    spead2.ThreadPool.set_affinity(1)
    config = spead2.send.StreamConfig(rate=0.0, max_heaps=2)
    if args.packet_size is not None:
        config.max_packet_size = args.packet_size
    stream = spead2.send.asyncio.UdpStream(thread_pool, [(args.host, args.port)], config)
    heap_size = args.heap_size
    item_group = spead2.send.ItemGroup()
    item_group.add_item(
        0x1600,
        "timestamp",
        "Index of the first sample",
        shape=(),
        format=[("u", spead2.Flavour().heap_address_bits)],
    )
    item_group.add_item(
        0x3300,
        "adc_samples",
        "ADC converter output",
        shape=(heap_size,),
        dtype=np.int8,
    )

    n_heaps = args.heaps
    states = [State(adc_samples=np.ones(heap_size, np.int8)) for _ in range(2)]
    start = time.perf_counter()
    for i in range(n_heaps):
        state = states[i % len(states)]
        await state.future  # Wait for any previous use of this state to complete
        state.adc_samples.fill(np.int_(i))
        item_group["timestamp"].value = i * heap_size
        item_group["adc_samples"].value = state.adc_samples
        heap = item_group.get_heap()
        state.future = stream.async_send_heap(heap)
    for state in states:
        await state.future
    elapsed = time.perf_counter() - start
    print(f"{heap_size * n_heaps / elapsed / 1e6:.2f} MB/s")

    await stream.async_send_heap(item_group.get_end())


if __name__ == "__main__":
    asyncio.run(main())
