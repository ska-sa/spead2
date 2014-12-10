#!/usr/bin/env python
import spead2.recv
import sys

items = []

def callback(heap):
    items.extend(heap.get_items())

with open('junkspeadfile', 'rb') as f:
    text = f.read()
stream = spead2.recv.Stream(2)
receiver = spead2.recv.Receiver()
receiver.add_buffer_reader(stream, text)
receiver.start()
while True:
    heap = stream.pop()
    if heap.cnt == 0:
        break
    items.extend(heap.get_items())
receiver.join()

del receiver
del stream
for item in items:
    print item.id, item.value
