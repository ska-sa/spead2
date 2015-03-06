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
    "localhost", 8888, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2, 1500, 1e7)
del thread_pool

shape = (40, 50)
item = spead2.Item(0x1234, 'foo', 'a foo item', shape=shape, dtype=np.int32)
item.value = np.zeros(shape, np.int32)
heap = spead2.send.Heap(1, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
heap.add_descriptor(item)
heap.add_item(item.id, item)
coro = stream.async_send_heap(heap)
# Delete things to check that there are no refcounting bugs
del heap
del stream
trollius.get_event_loop().run_until_complete(coro)
