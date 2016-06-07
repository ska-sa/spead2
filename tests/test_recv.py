#!/usr/bin/env python

# Copyright 2015 SKA South Africa
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division
import spead2
import spead2.recv
import sys
import logging

logging.basicConfig(level=logging.INFO)

items = []

thread_pool = spead2.ThreadPool()
stream = spead2.recv.Stream(thread_pool, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
del thread_pool
pool = spead2.MemoryPool(16384, 26214400, 12, 8)
stream.set_memory_allocator(pool)
if 0:
    with open('junkspeadfile', 'rb') as f:
        text = f.read()
    stream.add_buffer_reader(text)
else:
    stream.add_udp_reader(8888)

ig = spead2.ItemGroup()
num_heaps = 0
for heap in stream:
    print("Got heap", heap.cnt)
    items = ig.update(heap)
    for item in items.values():
        print(heap.cnt, item.name, item.value)
    num_heaps += 1
stream.stop()
print("Received", num_heaps, "heaps")
