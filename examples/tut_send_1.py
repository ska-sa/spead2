#!/usr/bin/env python3

import numpy as np

import spead2.send


def main():
    thread_pool = spead2.ThreadPool()
    config = spead2.send.StreamConfig(rate=100e6)
    stream = spead2.send.UdpStream(thread_pool, [("127.0.0.1", 8888)], config)
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
    for i in range(10):
        item_group["timestamp"].value = i * chunk_size
        item_group["adc_samples"].value = rng.integers(-100, 100, size=chunk_size, dtype=np.int8)
        heap = item_group.get_heap()
        stream.send_heap(heap)
    stream.send_heap(item_group.get_end())


if __name__ == "__main__":
    main()
