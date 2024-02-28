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

import numpy as np

import spead2.send


def main():
    thread_pool = spead2.ThreadPool()
    config = spead2.send.StreamConfig(rate=100e6)
    stream = spead2.send.UdpStream(thread_pool, [("127.0.0.1", 8888)], config)
    heap_size = 1024 * 1024
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

    rng = np.random.default_rng()
    for i in range(10):
        item_group["timestamp"].value = i * heap_size
        item_group["adc_samples"].value = rng.integers(-100, 100, size=heap_size, dtype=np.int8)
        heap = item_group.get_heap()
        stream.send_heap(heap)
    stream.send_heap(item_group.get_end())


if __name__ == "__main__":
    main()
