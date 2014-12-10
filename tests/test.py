#!/usr/bin/env python
import spead2.recv
import sys

items = []

stream = spead2.recv.Stream(2)
receiver = spead2.recv.Receiver()
if 0:
    with open('junkspeadfile', 'rb') as f:
        text = f.read()
    receiver.add_buffer_reader(stream, text)
else:
    receiver.add_udp_reader(stream, 8888)
receiver.start()

ig = spead2.recv.ItemGroup()
for heap in stream:
    ig.update(heap)
    for item in ig.items.itervalues():
        print item.name, item.value
receiver.stop()
