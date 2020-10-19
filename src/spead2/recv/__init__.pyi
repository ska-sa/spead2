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

from typing import Iterator, Any, List, Tuple, Sequence, Union, Text, Optional, ClassVar, overload
import socket

import spead2
from spead2 import _EndpointList

class RawItem:
    @property
    def id(self) -> int: ...
    @property
    def is_immediate(self) -> bool: ...
    @property
    def immediate_value(self) -> int: ...

class _HeapBase:
    @property
    def cnt(self) -> int: ...
    @property
    def flavour(self) -> spead2.Flavour: ...
    def get_items(self) -> List[RawItem]: ...
    def is_start_of_stream(self) -> bool: ...
    def is_end_of_stream(self) -> bool: ...

class Heap(_HeapBase):
    def get_descriptors(self) -> List[spead2.RawDescriptor]: ...

class IncompleteHeap(_HeapBase):
    @property
    def heap_length(self) -> int: ...
    @property
    def received_length(self) -> int: ...
    @property
    def payload_ranges(self) -> List[Tuple[int, int]]: ...

class StreamStats:
    heaps: int
    incomplete_heaps_evicted: int
    incomplete_Heaps_flushed: int
    packets: int
    batches: int
    worker_blocked: int
    max_batch: int
    single_packet_heaps: int
    search_dist: int
    def __add__(self, other: StreamStats) -> StreamStats: ...
    def __iadd__(self, other: StreamStats) -> None: ...

class StreamConfig:
    DEFAULT_MAX_HEAPS: ClassVar[int] = ...
    max_heaps: int
    bug_compat: int
    memcpy: int
    memory_allocator: spead2.MemoryAllocator
    stop_on_stop_item: bool
    allow_unsized_heaps: bool
    allow_out_of_order: bool
    def __init__(self, *, max_heaps: int = ..., bug_compat: int = ...,
                 memcpy: int = ..., memory_allocator: spead2.MemoryAllocator = ...,
                 stop_on_stop_item: bool = ..., allow_unsized_heaps: bool = ...,
                 allow_out_of_order: bool = ...) -> None: ...

class RingStreamConfig:
    DEFAULT_HEAPS: ClassVar[int]
    heaps: int
    contiguous_only: bool
    incomplete_keep_payload_ranges: bool
    def __init__(self, *, heaps: int = ..., contiguous_only: bool = ...,
                 incomplete_keep_payload_ranges: bool = ...) -> None: ...

class UdpIbvConfig:
    DEFAULT_BUFFER_SIZE: ClassVar[int]
    DEFAULT_MAX_SIZE: ClassVar[int]
    DEFAULT_MAX_POLL: ClassVar[int]

    endpoints: _EndpointList
    interface_address: str
    buffer_size: int
    max_size: int
    comp_vector: int
    max_poll: int

    def __init__(self, *, endpoints: _EndpointList = ..., interface_address: str = ...,
                 buffer_size: int = ..., max_size: int = ..., comp_vector: int = ...,
                 max_poll: int = ...) -> None: ...

class Ringbuffer:
    def size(self) -> int: ...
    def capacity(self) -> int: ...

# We make a dummy _Stream base class because mypy objects to the async stream
# type overloading get as a coroutine.
class _Stream:
    DEFAULT_UDP_IBV_MAX_SIZE: ClassVar[int]
    DEFAULT_UDP_IBV_BUFFER_SIZE: ClassVar[int]
    DEFAULT_UDP_IBV_MAX_POLL: ClassVar[int]
    DEFAULT_UDP_MAX_SIZE: ClassVar[int]
    DEFAULT_UDP_BUFFER_SIZE: ClassVar[int]
    DEFAULT_TCP_MAX_SIZE: ClassVar[int]
    DEFAULT_TCP_BUFFER_SIZE: ClassVar[int]

    def __init__(self, thread_pool: spead2.ThreadPool, config: StreamConfig = ...,
                 ring_config: RingStreamConfig = ...) -> None: ...
    def __iter__(self) -> Iterator[Heap]: ...
    def get_nowait(self) -> Heap: ...
    def add_buffer_reader(self, buffer: Any) -> None: ...
    @overload
    def add_udp_reader(self, port: int, max_size: int = ..., buffer_size: int = ...,
                       bind_hostname: str = ...) -> None: ...
    @overload
    def add_udp_reader(self, socket: socket.socket, max_size: int = ...) -> None: ...
    @overload
    def add_udp_reader(self, multicast_group: str, port: int, max_size: int = ...,
                       buffer_size: int = ..., interface_address: str = ...) -> None: ...
    @overload
    def add_udp_reader(self, multicast_group: str, port: int, max_size: int = ...,
                       buffer_size: int = ..., interface_index: int = ...) -> None: ...
    @overload
    def add_tcp_reader(self, port: int, max_size: int = ..., buffer_size: int = ...,
                       bind_hostname: str = ...) -> None: ...
    @overload
    def add_tcp_reader(self, acceptor: socket.socket, max_size: int = ...) -> None: ...
    @overload
    def add_udp_ibv_reader(self, multicast_group: str, port: int,
                           interface_address: str,
                           max_size: int = ..., buffer_size: int = ...,
                           comp_vector: int = ..., max_poll: int = ...) -> None: ...
    @overload
    def add_udp_ibv_reader(self, endpoints: Sequence[Tuple[str, int]],
                           interface_address: str,
                           max_size: int = ..., buffer_size: int = ...,
                           comp_vector: int = ..., max_poll: int = ...) -> None: ...
    @overload
    def add_udp_ibv_reader(self, config: UdpIbvConfig) -> None: ...
    def add_inproc_reader(self, queue: spead2.InprocQueue) -> None: ...
    def stop(self) -> None: ...
    @property
    def fd(self) -> int: ...
    @property
    def stats(self) -> StreamStats: ...
    @property
    def ringbuffer(self) -> Ringbuffer: ...
    @property
    def config(self) -> StreamConfig: ...
    @property
    def ring_config(self) -> RingStreamConfig: ...

class Stream(_Stream):
    def get(self) -> Heap: ...
