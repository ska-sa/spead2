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
    timestamps: np.ndarray
    heaps: spead2.send.HeapReferenceList
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
    batches = 2
    batch_heaps = max(1, 512 * 1024 // args.heap_size)
    config = spead2.send.StreamConfig(rate=0.0, max_heaps=batches * batch_heaps)
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

    def make_heaps(adc_samples, timestamps, first):
        heaps = []
        for j in range(batch_heaps):
            item_group["timestamp"].value = timestamps[j, ...]
            item_group["adc_samples"].value = adc_samples[j, ...]
            heap = item_group.get_heap(descriptors="none" if j or not first else "all", data="all")
            heaps.append(spead2.send.HeapReference(heap))
        return spead2.send.HeapReferenceList(heaps)

    states = []
    for _ in range(batches):
        adc_samples = np.ones((batch_heaps, heap_size), np.int8)
        timestamps = np.ones(batch_heaps, ">u8")
        states.append(
            State(
                adc_samples=adc_samples,
                timestamps=timestamps,
                heaps=make_heaps(adc_samples, timestamps, False),
            )
        )
    first_heaps = make_heaps(states[0].adc_samples, states[0].timestamps, True)

    start = time.perf_counter()
    for i in range(0, n_heaps, batch_heaps):
        state = states[(i // batch_heaps) % len(states)]
        end = min(i + batch_heaps, n_heaps)
        n = end - i
        await state.future  # Wait for any previous use of this state to complete
        state.adc_samples[:n] = np.arange(i, end).astype(np.int8)[:, np.newaxis]
        state.timestamps[:n] = np.arange(i * heap_size, end * heap_size, heap_size)
        heaps = state.heaps if i else first_heaps
        if n < batch_heaps:
            heaps = heaps[:n]
        state.future = stream.async_send_heaps(heaps, spead2.send.GroupMode.SERIAL)
    for state in states:
        await state.future
    elapsed = time.perf_counter() - start
    print(f"{heap_size * n_heaps / elapsed / 1e6:.2f} MB/s")

    await stream.async_send_heap(item_group.get_end())


if __name__ == "__main__":
    asyncio.run(main())
