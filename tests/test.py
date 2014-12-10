#!/usr/bin/env python
import spead2.recv
import sys

items = []

def callback(heap):
    items.extend(heap.get_items())

with open('junkspeadfile', 'rb') as f:
    text = f.read()
stream = spead2.recv.BufferStream(text)
stream.set_callback(callback)
stream.run()

for item in items:
    print item.id, item.value
