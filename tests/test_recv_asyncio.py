# Copyright 2018, 2020 National Research Foundation (SARAO)
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

import pytest

import spead2
import spead2.send
import spead2.recv.asyncio


@pytest.mark.asyncio
class TestRecvAsyncio:
    async def test_async_iter(self):
        tp = spead2.ThreadPool()
        queue = spead2.InprocQueue()
        sender = spead2.send.InprocStream(tp, [queue])
        ig = spead2.send.ItemGroup()
        sender.send_heap(ig.get_start())
        sender.send_heap(ig.get_end())
        queue.stop()
        heaps = []
        recv_config = spead2.recv.StreamConfig(stop_on_stop_item=False)
        receiver = spead2.recv.asyncio.Stream(tp, recv_config)
        receiver.add_inproc_reader(queue)
        async for heap in receiver:
            heaps.append(heap)
        assert len(heaps) == 2
        assert heaps[0].is_start_of_stream()
        assert heaps[1].is_end_of_stream()
