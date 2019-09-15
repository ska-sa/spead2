# Copyright 2018 SKA South Africa
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

import asynctest
from nose.tools import assert_equal, assert_true

import spead2
import spead2.send
import spead2.recv.asyncio


class TestRecvAsyncio(asynctest.TestCase):
    async def test_async_iter(self):
        tp = spead2.ThreadPool()
        queue = spead2.InprocQueue()
        sender = spead2.send.InprocStream(tp, queue)
        ig = spead2.send.ItemGroup()
        sender.send_heap(ig.get_start())
        sender.send_heap(ig.get_end())
        queue.stop()
        heaps = []
        receiver = spead2.recv.asyncio.Stream(tp)
        receiver.stop_on_stop_item = False
        receiver.add_inproc_reader(queue)
        async for heap in receiver:
            heaps.append(heap)
        assert_equal(2, len(heaps))
        assert_true(heaps[0].is_start_of_stream())
        assert_true(heaps[1].is_end_of_stream())
