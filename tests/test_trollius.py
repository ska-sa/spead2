#!/usr/bin/env python
import spead2
import spead2.recv
import spead2.recv.trollius
import sys
import logging
import trollius
from trollius import From

def run(stream):
    ig = spead2.recv.ItemGroup()
    try:
        while True:
            heap = yield From(stream.get())
            print "Got heap", heap.cnt
            ig.update(heap)
            for item in ig.items.itervalues():
                print heap.cnt, item.name, item.value.shape
    except spead2.Stopped:
        pass

logging.basicConfig(level=logging.DEBUG)

items = []

receiver = spead2.recv.Receiver()
coros = []
pool = spead2.Mempool(16384, 26214400, 32, 16)
for i in range(4):
    stream = spead2.recv.trollius.Stream(spead2.BUG_COMPAT_PYSPEAD_0_5_2)
    stream.set_mempool(pool)
    receiver.add_udp_reader(stream, 8888 + i)
    coros.append(run(stream))
receiver.start(4)

trollius.get_event_loop().run_until_complete(trollius.wait(coros))
