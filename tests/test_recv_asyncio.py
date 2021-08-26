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

import asyncio

import pytest

import spead2
import spead2.send
import spead2.recv.asyncio


pytestmark = [pytest.mark.asyncio]


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


class MyChunk(spead2.recv.Chunk):
    """Subclasses Chunk to carry extra metadata."""

    def __init__(self, label, **kwargs):
        kwargs.setdefault('data', bytearray(10))
        kwargs.setdefault('present', bytearray(1))
        super().__init__(**kwargs)
        self.label = label


class TestChunkRingbuffer:
    @pytest.fixture
    def chunk_ringbuffer(self):
        return spead2.recv.asyncio.ChunkRingbuffer(3)

    async def test_get_ready(self, chunk_ringbuffer):
        chunk_ringbuffer.put_nowait(MyChunk("a"))
        chunk = await chunk_ringbuffer.get()
        assert chunk.label == "a"

    async def test_get_block(self, chunk_ringbuffer):
        task = asyncio.get_event_loop().create_task(chunk_ringbuffer.get())
        await asyncio.sleep(0.01)
        chunk_ringbuffer.put_nowait(MyChunk("a"))
        chunk = await task
        assert chunk.label == "a"

    async def test_get_cancel(self, chunk_ringbuffer):
        task1 = asyncio.get_event_loop().create_task(chunk_ringbuffer.get())
        task2 = asyncio.get_event_loop().create_task(chunk_ringbuffer.get())
        await asyncio.sleep(0.01)
        task1.cancel()
        chunk_ringbuffer.put_nowait(MyChunk("a"))
        chunk = await task2
        assert chunk.label == "a"

    async def test_get_stop(self, chunk_ringbuffer):
        task = asyncio.get_event_loop().create_task(chunk_ringbuffer.get())
        await asyncio.sleep(0.01)
        chunk_ringbuffer.stop()
        with pytest.raises(spead2.Stopped):
            await task
        with pytest.raises(spead2.Stopped):
            await chunk_ringbuffer.get()

    async def test_put_ready(self, chunk_ringbuffer):
        await chunk_ringbuffer.put(MyChunk("a"))
        chunk = chunk_ringbuffer.get_nowait()
        assert chunk.label == "a"

    async def test_put_block(self, chunk_ringbuffer):
        while not chunk_ringbuffer.full():
            chunk_ringbuffer.put_nowait(MyChunk("filler"))
        task = asyncio.get_event_loop().create_task(chunk_ringbuffer.put(MyChunk("a")))
        await asyncio.sleep(0.01)
        for i in range(chunk_ringbuffer.maxsize):
            chunk = chunk_ringbuffer.get_nowait()
            assert chunk.label == "filler"
        await task
        chunk = chunk_ringbuffer.get_nowait()
        assert chunk.label == "a"

    async def test_put_stop(self, chunk_ringbuffer):
        while not chunk_ringbuffer.full():
            chunk_ringbuffer.put_nowait(MyChunk("filler"))
        task = asyncio.get_event_loop().create_task(chunk_ringbuffer.put(MyChunk("a")))
        await asyncio.sleep(0.01)
        chunk_ringbuffer.stop()
        with pytest.raises(spead2.Stopped):
            await task
        for i in range(chunk_ringbuffer.maxsize):
            chunk = chunk_ringbuffer.get_nowait()
            assert chunk.label == "filler"
        with pytest.raises(spead2.Stopped):
            chunk_ringbuffer.get_nowait()
        with pytest.raises(spead2.Stopped):
            await chunk_ringbuffer.put(MyChunk("b"))
