# Copyright 2019-2021, 2023-2024 National Research Foundation (SARAO)
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

import enum
import socket
from collections.abc import Iterator, Sequence
from typing import ClassVar, overload

import spead2
from spead2 import _EndpointList

class Heap:
    def __init__(self, flavour: spead2.Flavour) -> None: ...
    @property
    def flavour(self) -> spead2.Flavour: ...
    @property
    def repeat_pointers(self) -> bool: ...
    @repeat_pointers.setter
    def repeat_pointers(self, value: bool) -> None: ...
    def add_item(self, item: spead2.Item) -> None: ...
    def add_descriptor(self, descriptor: spead2.Descriptor) -> None: ...
    def add_start(self) -> None: ...
    def add_end(self) -> None: ...

class HeapReference:
    cnt: int
    substream_index: int
    rate: float

    def __init__(
        self, heap: Heap, *, cnt: int = ..., substream_index: int = ..., rate: float = ...
    ) -> None: ...
    @property
    def heap(self) -> Heap: ...

class HeapReferenceList:
    def __init__(self, heaps: list[HeapReference]) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, _index: slice) -> HeapReferenceList: ...

class PacketGenerator:
    def __init__(self, heap: Heap, cnt: int, max_packet_size: int) -> None: ...
    def __iter__(self) -> Iterator[bytes]: ...

class RateMethod(enum.Enum):
    SW = ...
    HW = ...
    AUTO = ...

class StreamConfig:
    DEFAULT_MAX_PACKET_SIZE: ClassVar[int]
    DEFAULT_MAX_HEAPS: ClassVar[int]
    DEFAULT_BURST_SIZE: ClassVar[int]
    DEFAULT_BURST_RATE_RATIO: ClassVar[float]
    DEFAULT_RATE_METHOD: ClassVar[RateMethod]

    max_packet_size: int
    rate: float
    burst_size: int
    max_heaps: int
    burst_rate_ratio: float
    rate_method: RateMethod

    def __init__(
        self,
        *,
        max_packet_size: int = ...,
        rate: float = ...,
        burst_size: int = ...,
        max_heaps: int = ...,
        burst_rate_ratio: float = ...,
        rate_method: RateMethod = ...,
    ) -> None: ...
    @property
    def burst_rate(self) -> float: ...

class GroupMode(enum.Enum):
    ROUND_ROBIN = ...
    SERIAL = ...

class Stream:
    def set_cnt_sequence(self, next: int, step: int) -> None: ...
    @property
    def num_substreams(self) -> int: ...

class SyncStream(Stream):
    def send_heap(
        self, heap: Heap, cnt: int = ..., substream_index: int = ..., rate: float = ...
    ) -> None: ...
    def send_heaps(
        self, heaps: list[HeapReference] | HeapReferenceList, mode: GroupMode
    ) -> None: ...

class _UdpStream:
    DEFAULT_BUFFER_SIZE: ClassVar[int]

    # There are a lot of overloads listed here, because the bindings have
    # overloads where required arguments follow optional arguments, and that
    # can't be represented in pure Python. So we have to distinguish the
    # cases where the required arguments are specified by name versus
    # position.
    @overload
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        endpoints: _EndpointList,
        config: StreamConfig = ...,
        buffer_size: int = ...,
        interface_address: str = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        endpoints: _EndpointList,
        config: StreamConfig,
        buffer_size: int,
        ttl: int,
    ) -> None: ...
    @overload
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        endpoints: _EndpointList,
        config: StreamConfig = ...,
        buffer_size: int = ...,
        *,
        ttl: int,
    ) -> None: ...
    @overload
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        endpoints: _EndpointList,
        config: StreamConfig,
        buffer_size: int,
        ttl: int,
        interface_address: str,
    ) -> None: ...
    @overload
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        endpoints: _EndpointList,
        config: StreamConfig = ...,
        buffer_size: int = ...,
        *,
        ttl: int,
        interface_address: str,
    ) -> None: ...
    @overload
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        endpoints: _EndpointList,
        config: StreamConfig,
        buffer_size: int,
        ttl: int,
        interface_index: int,
    ) -> None: ...
    @overload
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        endpoints: _EndpointList,
        config: StreamConfig = ...,
        buffer_size: int = ...,
        *,
        ttl: int,
        interface_index: int,
    ) -> None: ...
    @overload
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        socket: socket.socket,
        endpoints: _EndpointList,
        config: StreamConfig = ...,
    ) -> None: ...

class UdpStream(_UdpStream, SyncStream): ...

class UdpIbvConfig:
    DEFAULT_BUFFER_SIZE: ClassVar[int]
    DEFAULT_MAX_POLL: ClassVar[int]

    endpoints: _EndpointList
    interface_address: str
    buffer_size: int
    ttl: int
    comp_vector: int
    max_poll: int
    memory_regions: list

    def __init__(
        self,
        *,
        endpoints: _EndpointList = ...,
        interface_address: str = ...,
        buffer_size: int = ...,
        ttl: int = ...,
        comp_vector: int = ...,
        max_poll: int = ...,
        memory_regions: list = ...,
    ) -> None: ...

class _UdpIbvStream:
    def __init__(
        self, thread_pool: spead2.ThreadPool, config: StreamConfig, udp_ibv_config: UdpIbvConfig
    ) -> None: ...

class UdpIbvStream(_UdpIbvStream, SyncStream): ...

class _TcpStream:
    DEFAULT_BUFFER_SIZE: ClassVar[int]

class TcpStream(_TcpStream, SyncStream):
    @overload
    def __init__(
        self, thread_pool: spead2.ThreadPool, socket: socket.socket, config: StreamConfig = ...
    ) -> None: ...
    @overload
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        endpoints: _EndpointList,
        config: StreamConfig = ...,
        buffer_size: int = ...,
        interface_address: str = ...,
    ) -> None: ...

class BytesStream(SyncStream):
    def getvalue(self) -> bytes: ...
    def __init__(self, thread_pool: spead2.ThreadPool, config: StreamConfig = ...) -> None: ...

class _InprocStream:
    @property
    def queues(self) -> Sequence[spead2.InprocQueue]: ...
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        queues: list[spead2.InprocQueue],
        config: StreamConfig = ...,
    ) -> None: ...

class InprocStream(_InprocStream, SyncStream): ...

class HeapGenerator:
    def __init__(
        self,
        item_group: spead2.ItemGroup,
        descriptor_frequency: int | None = None,
        flavour: spead2.Flavour = ...,
    ) -> None: ...
    def add_to_heap(self, heap: Heap, descriptors: str = ..., data: str = ...) -> Heap: ...
    def get_heap(self, descriptors: str = ..., data: str = ...) -> Heap: ...
    def get_start(self) -> Heap: ...
    def get_end(self) -> Heap: ...

class ItemGroup(spead2.ItemGroup, HeapGenerator):
    def __init__(
        self, descriptor_frequency: int | None = None, flavour: spead2.Flavour = ...
    ) -> None: ...
