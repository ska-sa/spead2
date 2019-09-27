# Copyright 2015, 2019 SKA South Africa
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

import asyncio

import numpy as np
from nose.tools import assert_greater, assert_equal, assert_raises

import spead2
import spead2.send
import spead2.send.asyncio
from spead2.send.asyncio import UdpStream


class TestUdpStream:
    def setup(self):
        # Make a stream slow enough that we can test async interactions
        config = spead2.send.StreamConfig(rate=5e6)
        self.stream = UdpStream(spead2.ThreadPool(), 'localhost', 8888, config)
        self.ig = spead2.send.ItemGroup()
        self.ig.add_item(0x1000, 'test', 'Test item', shape=(256 * 1024,), dtype=np.uint8)
        self.ig['test'].value = np.zeros((256 * 1024,), np.uint8)
        self.heap = self.ig.get_heap()

    async def _test_async_flush(self):
        assert_greater(self.stream._active, 0)
        await self.stream.async_flush()
        assert_equal(self.stream._active, 0)

    def test_async_flush(self):
        for i in range(3):
            asyncio.ensure_future(self.stream.async_send_heap(self.heap))
        # The above only queues up the async sends on the event loop. The rest of the
        # test needs to be run from inside the event loop
        asyncio.get_event_loop().run_until_complete(self._test_async_flush())

    def test_async_flush_fail(self):
        """Test async_flush in the case that the last heap sent failed.
        This is arranged by filling up the queue slots first.
        """
        for i in range(5):
            asyncio.ensure_future(self.stream.async_send_heap(self.heap))
        # The above only queues up the async sends on the event loop. The rest of the
        # test needs to be run from inside the event loop
        asyncio.get_event_loop().run_until_complete(self._test_async_flush())

    async def _test_send_error(self, future):
        with assert_raises(IOError):
            await future

    def test_send_error(self):
        """An error in sending must be reported through the future."""
        # Create a stream with a packet size that is bigger than the likely
        # MTU. It should cause an error.
        stream = UdpStream(
            spead2.ThreadPool(), "localhost", 8888,
            spead2.send.StreamConfig(max_packet_size=100000), buffer_size=0)
        future = stream.async_send_heap(self.heap)
        asyncio.get_event_loop().run_until_complete(self._test_send_error(future))


class TestTcpStream:
    async def _test_connect_failed(self):
        thread_pool = spead2.ThreadPool()
        with assert_raises(IOError):
            await spead2.send.asyncio.TcpStream.connect(thread_pool, '127.0.0.1', 8887)

    def test_connect_failed(self):
        asyncio.get_event_loop().run_until_complete(self._test_connect_failed())
