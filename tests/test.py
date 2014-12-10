#!/usr/bin/env python
import spead2.recv
import sys

items = []

stream = spead2.recv.Stream(2)
receiver = spead2.recv.Receiver()
if False:
    with open('junkspeadfile', 'rb') as f:
        text = f.read()
    receiver.add_buffer_reader(stream, text)
else:
    receiver.add_udp_reader(stream, 8888)
receiver.start()
for heap in stream:
    heap_items = heap.get_items()
    for item in heap_items:
        print item.id, item.value
    items.extend(heap_items)
receiver.join()

del receiver
del stream
for item in items:
    print item.id, item.value
