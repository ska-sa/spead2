# Copyright 2015, 2019-2024 National Research Foundation (SARAO)
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
import gc
import logging
import weakref

import numpy as np
import pytest

import spead2
import spead2.send
import spead2.send.asyncio
from spead2.send.asyncio import UdpStream


@pytest.mark.asyncio
class TestUdpStream:
    @pytest.fixture
    def stream(self, unused_udp_port):
        # Make a stream slow enough that we can test async interactions
        config = spead2.send.StreamConfig(rate=5e6)
        return UdpStream(spead2.ThreadPool(), [("localhost", unused_udp_port)], config)

    @pytest.fixture
    def ig(self):
        ig = spead2.send.ItemGroup()
        ig.add_item(0x1000, "test", "Test item", shape=(256 * 1024,), dtype=np.uint8)
        ig["test"].value = np.zeros((256 * 1024,), np.uint8)
        return ig

    @pytest.fixture
    def heap(self, ig):
        return ig.get_heap()

    async def _test_async_flush(self, stream):
        assert stream._active > 0
        await stream.async_flush()
        assert stream._active == 0

    async def test_async_flush(self, stream, heap):
        send_heap_futures = []
        for i in range(3):
            send_heap_futures.append(asyncio.ensure_future(stream.async_send_heap(heap)))
        await self._test_async_flush(stream)
        await asyncio.gather(*send_heap_futures)

    async def test_async_flush_fail(self, stream, heap):
        """Test async_flush in the case that the last heap sent failed.

        This is arranged by filling up the queue slots first.
        """
        send_heap_futures = []
        for i in range(5):
            send_heap_futures.append(asyncio.ensure_future(stream.async_send_heap(heap)))
        await self._test_async_flush(stream)
        with pytest.raises(BlockingIOError):
            await asyncio.gather(*send_heap_futures)

    async def test_send_error(self, unused_udp_port, heap):
        """An error in sending must be reported through the future."""
        # Create a stream with a packet size that is bigger than the likely
        # MTU. It should cause an error.
        stream = UdpStream(
            spead2.ThreadPool(),
            [("localhost", unused_udp_port)],
            spead2.send.StreamConfig(max_packet_size=100000),
            buffer_size=0,
        )
        with pytest.raises(IOError):
            await stream.async_send_heap(heap)

    async def test_async_send_heap_refcount(self, stream, ig):
        """async_send_heap must release the reference to the heap."""
        heap = ig.get_heap()
        weak = weakref.ref(heap)
        future = stream.async_send_heap(weak())
        heap = None
        await future
        gc.collect()
        assert weak() is None

    async def test_async_send_heaps_refcount(self, stream, ig):
        """async_send_heaps must release the reference to the heap."""
        heap = ig.get_heap()
        weak = weakref.ref(heap)
        future = stream.async_send_heaps(
            [spead2.send.HeapReference(weak())], spead2.send.GroupMode.ROUND_ROBIN
        )
        heap = None
        await future
        for i in range(5):  # Try extra hard to make PyPy release things
            gc.collect()
        assert weak() is None

    async def test_cancel(self, caplog, stream, heap):
        """Cancelling the future must work gracefully."""
        with caplog.at_level(logging.ERROR):
            future = stream.async_send_heap(heap)
            future.cancel()
            with pytest.raises(asyncio.CancelledError):
                await future
            # Send another heap to ensure that process_callbacks has time to run.
            await stream.async_send_heap(heap)
        # An exception in process_callbacks doesn't propagate anywhere we can
        # easily access it, but it does cause the event loop to log an error.
        assert not caplog.records


@pytest.mark.asyncio
class TestTcpStream:
    async def test_connect_failed(self, unused_tcp_port):
        thread_pool = spead2.ThreadPool()
        with pytest.raises(IOError):
            await spead2.send.asyncio.TcpStream.connect(
                thread_pool, [("127.0.0.1", unused_tcp_port)]
            )
