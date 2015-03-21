#!/usr/bin/env python
import spead2
import spead2.send
import spead2.send.trollius
import sys
import logging
import numpy as np
import trollius

logging.basicConfig(level=logging.INFO)

thread_pool = spead2.ThreadPool()
stream = spead2.send.trollius.UdpStream(thread_pool,
    "localhost", 8888, spead2.send.StreamConfig(max_packet_size=1500, rate=1e7))
del thread_pool  # Make sure this doesn't crash anything

shape = (40, 50)
ig = spead2.send.ItemGroup(heap_address_bits=48, bug_compat=spead2.BUG_COMPAT_PYSPEAD_0_5_2)
item = ig.add_item(0x1234, 'foo', 'a foo item', shape=shape, dtype=np.int32)
item.value = np.zeros(shape, np.int32)
coros = [
    stream.async_send_heap(ig.get_heap()),
    stream.async_send_heap(ig.get_end())
]
# Delete things to check that there are no refcounting bugs
del ig
del stream
trollius.get_event_loop().run_until_complete(trollius.wait(coros))
