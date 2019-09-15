# Copyright 2018-2019 SKA South Africa
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

"""Tests that data can be passed through the various async transports"""
import socket
import asyncio

import spead2
import spead2.send
import spead2.recv.asyncio
import spead2.send.asyncio

from . import test_passthrough


class BaseTestPassthroughAsync(test_passthrough.BaseTestPassthrough):
    def transmit_item_group(self, item_group, memcpy, allocator, new_order='='):
        self.loop = asyncio.new_event_loop()
        ret = self.loop.run_until_complete(
            self.transmit_item_group_async(item_group, memcpy, allocator, new_order))
        self.loop.close()
        return ret

    async def transmit_item_group_async(self, item_group, memcpy, allocator, new_order='='):
        thread_pool = spead2.ThreadPool(2)
        receiver = spead2.recv.asyncio.Stream(thread_pool, loop=self.loop)
        receiver.set_memcpy(memcpy)
        if allocator is not None:
            receiver.set_memory_allocator(allocator)
        await self.prepare_receiver(receiver)
        sender = await self.prepare_sender(thread_pool)
        gen = spead2.send.HeapGenerator(item_group)
        await sender.async_send_heap(gen.get_heap())
        await sender.async_send_heap(gen.get_end())
        await sender.async_flush()
        received_item_group = spead2.ItemGroup()
        while True:
            try:
                heap = await receiver.get()
            except spead2.Stopped:
                break
            else:
                received_item_group.update(heap, new_order)
        return received_item_group


class TestPassthroughUdp(BaseTestPassthroughAsync):
    async def prepare_receiver(self, receiver):
        receiver.add_udp_reader(8888, bind_hostname="localhost")

    async def prepare_sender(self, thread_pool):
        return spead2.send.asyncio.UdpStream(
            thread_pool, "localhost", 8888,
            spead2.send.StreamConfig(rate=1e7),
            buffer_size=0, loop=self.loop)


class TestPassthroughUdpCustomSocket(BaseTestPassthroughAsync):
    async def prepare_receiver(self, receiver):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.bind(('127.0.0.1', 0))
        self._port = sock.getsockname()[1]
        receiver.add_udp_reader(sock)
        sock.close()

    async def prepare_sender(self, thread_pool):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        stream = spead2.send.asyncio.UdpStream(
            thread_pool, sock, '127.0.0.1', self._port,
            spead2.send.StreamConfig(rate=1e7), loop=self.loop)
        sock.close()
        return stream


class TestPassthroughTcp(BaseTestPassthroughAsync):
    async def prepare_receiver(self, receiver):
        receiver.add_tcp_reader(8888, bind_hostname="127.0.0.1")

    async def prepare_sender(self, thread_pool):
        sender = await spead2.send.asyncio.TcpStream.connect(
            thread_pool, "127.0.0.1", 8888, loop=self.loop)
        return sender


class TestPassthroughTcpCustomSocket(BaseTestPassthroughAsync):
    async def prepare_receiver(self, receiver):
        sock = socket.socket()
        # Prevent second iteration of the test from failing
        sock.bind(('127.0.0.1', 0))
        self._port = sock.getsockname()[1]
        sock.listen(1)
        receiver.add_tcp_reader(sock)
        sock.close()

    async def prepare_sender(self, thread_pool):
        sock = socket.socket()
        sock.setblocking(False)
        await self.loop.sock_connect(sock, ('127.0.0.1', self._port))
        sender = spead2.send.asyncio.TcpStream(
            thread_pool, sock, loop=self.loop)
        sock.close()
        return sender


class TestPassthroughInproc(BaseTestPassthroughAsync):
    async def prepare_receiver(self, receiver):
        receiver.add_inproc_reader(self._queue)

    async def prepare_sender(self, thread_pool):
        return spead2.send.asyncio.InprocStream(thread_pool, self._queue, loop=self.loop)

    async def transmit_item_group_async(self, item_group, memcpy, allocator, new_order='='):
        self._queue = spead2.InprocQueue()
        ret = await super(TestPassthroughInproc, self).transmit_item_group_async(
            item_group, memcpy, allocator, new_order)
        self._queue.stop()
        return ret
