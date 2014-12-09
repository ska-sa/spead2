#!/usr/bin/env python
import spead2
import sys

def callback(heap):
    print heap
    #print heap.get_items()
    item = heap.get_items()[0]
    del heap
    print item.value

with open('junkspeadfile', 'rb') as f:
    text = f.read()
stream = spead2.BufferStream(text)
stream.set_callback(callback)
stream.run()
