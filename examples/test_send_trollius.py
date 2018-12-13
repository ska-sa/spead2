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

import spead2
import spead2.send
import spead2.send.trollius
import logging
import numpy as np
import trollius

logging.basicConfig(level=logging.INFO)

thread_pool = spead2.ThreadPool()
stream = spead2.send.trollius.UdpStream(
    thread_pool, "127.0.0.1", 8888, spead2.send.StreamConfig(rate=1e7))
del thread_pool  # Make sure this doesn't crash anything

shape = (40, 50)
ig = spead2.send.ItemGroup(flavour=spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2))
item = ig.add_item(0x1234, 'foo', 'a foo item', shape=shape, dtype=np.int32)
item.value = np.zeros(shape, np.int32)
futures = [
    stream.async_send_heap(ig.get_heap()),
    stream.async_send_heap(ig.get_end())
]
# Delete things to check that there are no refcounting bugs
del ig
del stream
trollius.get_event_loop().run_until_complete(trollius.wait(futures))
