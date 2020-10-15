#!/usr/bin/env python

# Copyright 2015, 2020 National Research Foundation (SARAO)
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

import spead2
import spead2.recv
import logging

logging.basicConfig(level=logging.INFO)

thread_pool = spead2.ThreadPool()
stream = spead2.recv.Stream(
    thread_pool,
    spead2.recv.StreamConfig(memory_allocator=spead2.MemoryPool(16384, 26214400, 12, 8))
)
del thread_pool
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
