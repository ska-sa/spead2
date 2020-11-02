# Copyright 2019-2020 National Research Foundation (SARAO)
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
import socket
from typing import List, Optional, overload

import spead2
import spead2.send

from spead2 import _EndpointList

class _AsyncStream(spead2.send._Stream):
    @property
    def fd(self) -> int: ...
    def flush(self) -> None: ...
    def async_send_heap(self, heap: spead2.send.Heap, cnt: int = ...,
                        substream_index: int = ...) -> asyncio.Future[int]: ...
    def async_send_heaps(self,
                         heaps: List[spead2.send.HeapReference],
                         mode: spead2.send.GroupMode) -> asyncio.Future[int]: ...
    async def async_flush(self) -> None: ...

class UdpStream(spead2.send._UdpStream, _AsyncStream):
    pass

class UdpIbvStream(spead2.send._UdpIbvStream, _AsyncStream):
    pass

class TcpStream(spead2.send._TcpStream, _AsyncStream):
    def __init__(self, thread_pool: spead2.ThreadPool, socket: socket.socket,
                 config: spead2.send.StreamConfig = ...) -> None: ...
    @overload
    @classmethod
    async def connect(self, thread_pool: spead2.ThreadPool,
                      hostname: str, port: int,
                      config: spead2.send.StreamConfig = ...,
                      buffer_size: int = ..., interface_address: str = ...) -> None: ...
    @overload
    @classmethod
    async def connect(self, thread_pool: spead2.ThreadPool,
                      endpoints: _EndpointList,
                      config: spead2.send.StreamConfig = ...,
                      buffer_size: int = ..., interface_address: str = ...) -> None: ...


class InprocStream(spead2.send._InprocStream, _AsyncStream):
    pass
