#!/usr/bin/env python
import spead2
import spead2.recv
import sys
import logging

logging.basicConfig(level=logging.INFO)

items = []

thread_pool = spead2.ThreadPool()
stream = spead2.recv.Stream(thread_pool, spead2.BUG_COMPAT_PYSPEAD_0_5_2, 8)
del thread_pool
pool = spead2.MemPool(16384, 26214400, 12, 8)
stream.set_mem_pool(pool)
if 0:
    with open('junkspeadfile', 'rb') as f:
        text = f.read()
    stream.add_buffer_reader(text)
else:
    stream.add_udp_reader(8888)

ig = spead2.recv.ItemGroup()
num_heaps = 0
for heap in stream:
    print "Got heap", heap.cnt
    ig.update(heap)
    for item in ig.items.itervalues():
        print heap.cnt, item.name, item.value.shape
    num_heaps += 1
stream.stop()
print "Received", num_heaps, "heaps"
