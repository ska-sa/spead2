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

from __future__ import division, print_function
import trollius
from trollius import From, Return
import numpy as np
import spead2
import spead2.send
import spead2.send.trollius
from spead2.send.trollius import UdpStream
from nose.tools import *


class TestUdpStream(object):
    def setup(self):
        # Make a stream slow enough that we can test async interactions
        config = spead2.send.StreamConfig(rate=5e6)
        self.stream = UdpStream(spead2.ThreadPool(), 'localhost', 8888, config)
        self.ig = spead2.send.ItemGroup()
        self.ig.add_item(0x1000, 'test', 'Test item', shape=(256 * 1024,), dtype=np.uint8)
        self.ig['test'].value = np.zeros((256 * 1024,), np.uint8)
        self.heap = self.ig.get_heap()

    @trollius.coroutine
    def _test_async_flush(self):
        assert_greater(self.stream._active, 0)
        yield From(self.stream.async_flush())
        assert_equal(self.stream._active, 0)

    def test_async_flush(self):
        for i in range(3):
            trollius.async(self.stream.async_send_heap(self.heap))
        # The above only queues up the async sends on the event loop. The rest of the
        # test needs to be run from inside the event loop
        trollius.get_event_loop().run_until_complete(self._test_async_flush())
