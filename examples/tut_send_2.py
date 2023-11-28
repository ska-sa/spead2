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

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

import spead2.send
import spead2.send.asyncio


@dataclass
class State:
    future: asyncio.Future[int] = field(default_factory=asyncio.Future)


async def main():
    thread_pool = spead2.ThreadPool()
    config = spead2.send.StreamConfig(rate=0.0, max_heaps=2)
    stream = spead2.send.asyncio.UdpStream(thread_pool, [("127.0.0.1", 8888)], config)
    chunk_size = 1024 * 1024
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
        shape=(chunk_size,),
        dtype=np.int8,
    )

    rng = np.random.default_rng()
    n_heaps = 100
    start = time.monotonic()
    old_state = None
    for i in range(n_heaps):
        new_state = State()
        item_group["timestamp"].value = i * chunk_size
        item_group["adc_samples"].value = rng.integers(-100, 100, size=chunk_size, dtype=np.int8)
        heap = item_group.get_heap()
        new_state.future = stream.async_send_heap(heap)
        if old_state is not None:
            await old_state.future
        old_state = new_state
    await old_state.future
    elapsed = time.monotonic() - start
    print(f"{chunk_size * n_heaps / elapsed / 1e6:.2f} MB/s")
    await stream.async_send_heap(item_group.get_end())


if __name__ == "__main__":
    asyncio.run(main())
