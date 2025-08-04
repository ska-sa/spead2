# Copyright 2019-2023 National Research Foundation (SARAO)
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

import collections.abc
import enum
import socket
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, ClassVar, overload

from typing_extensions import Self

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
    def get_items(self) -> list[RawItem]: ...
    def is_start_of_stream(self) -> bool: ...
    def is_end_of_stream(self) -> bool: ...

class Heap(_HeapBase):
    def get_descriptors(self) -> list[spead2.RawDescriptor]: ...

class IncompleteHeap(_HeapBase):
    @property
    def heap_length(self) -> int: ...
    @property
    def received_length(self) -> int: ...
    @property
    def payload_ranges(self) -> list[tuple[int, int]]: ...

class StreamStatConfig:
    class Mode(enum.Enum):
        COUNTER: int = ...
        MAXIMUM: int = ...
    def __init__(self, name: str, mode: StreamStatConfig.Mode = ...) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def mode(self) -> StreamStatConfig.Mode: ...
    def combine(self, a: int, b: int) -> int: ...
    # __eq__ and __ne__ not listed because they're already defined for object

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
    @property
    def config(self) -> list[StreamStatConfig]: ...
    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, name: str) -> int: ...
    @overload
    def __setitem__(self, index: int, value: int) -> None: ...
    @overload
    def __setitem__(self, name: str, value: int) -> None: ...
    def __contains__(self, name: str) -> bool: ...
    def get(self, _name: str, _default: Any = None) -> Any: ...
    def items(self) -> Iterable[tuple[str, int]]: ...
    def __iter__(self) -> Iterator[str]: ...
    def keys(self) -> Iterable[str]: ...
    def values(self) -> Iterable[int]: ...
    def __len__(self) -> int: ...
    def __add__(self, other: StreamStats) -> StreamStats: ...
    def __iadd__(self, other: StreamStats) -> Self: ...

class StreamConfig:
    DEFAULT_MAX_HEAPS: ClassVar[int] = ...
    max_heaps: int
    substreams: int
    bug_compat: int
    memcpy: int
    memory_allocator: spead2.MemoryAllocator
    stop_on_stop_item: bool
    allow_unsized_heaps: bool
    allow_out_of_order: bool
    stream_id: int
    explicit_start: bool
    @property
    def stats(self) -> list[StreamStatConfig]: ...
    def __init__(
        self,
        *,
        max_heaps: int = ...,
        substreams: int = ...,
        bug_compat: int = ...,
        memcpy: int = ...,
        memory_allocator: spead2.MemoryAllocator = ...,
        stop_on_stop_item: bool = ...,
        allow_unsized_heaps: bool = ...,
        allow_out_of_order: bool = ...,
        stream_id: int = ...,
        explicit_start: bool = ...,
    ) -> None: ...
    def add_stat(self, name: str, mode: StreamStatConfig.Mode = ...) -> int: ...
    def get_stat_index(self, name: str) -> int: ...
    def next_stat_index(self) -> int: ...

class RingStreamConfig:
    DEFAULT_HEAPS: ClassVar[int]
    heaps: int
    contiguous_only: bool
    incomplete_keep_payload_ranges: bool
    def __init__(
        self,
        *,
        heaps: int = ...,
        contiguous_only: bool = ...,
        incomplete_keep_payload_ranges: bool = ...,
    ) -> None: ...

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

    def __init__(
        self,
        *,
        endpoints: _EndpointList = ...,
        interface_address: str = ...,
        buffer_size: int = ...,
        max_size: int = ...,
        comp_vector: int = ...,
        max_poll: int = ...,
    ) -> None: ...

class Ringbuffer:
    def size(self) -> int: ...
    def capacity(self) -> int: ...

class _Stream:
    DEFAULT_UDP_MAX_SIZE: ClassVar[int]
    DEFAULT_UDP_BUFFER_SIZE: ClassVar[int]
    DEFAULT_TCP_MAX_SIZE: ClassVar[int]
    DEFAULT_TCP_BUFFER_SIZE: ClassVar[int]

    def add_buffer_reader(self, buffer: Any) -> None: ...
    @overload
    def add_udp_reader(
        self, port: int, max_size: int = ..., buffer_size: int = ..., bind_hostname: str = ...
    ) -> None: ...
    @overload
    def add_udp_reader(self, socket: socket.socket, max_size: int = ...) -> None: ...
    @overload
    def add_udp_reader(
        self,
        multicast_group: str,
        port: int,
        max_size: int = ...,
        buffer_size: int = ...,
        interface_address: str = ...,
    ) -> None: ...
    @overload
    def add_udp_reader(
        self,
        multicast_group: str,
        port: int,
        max_size: int = ...,
        buffer_size: int = ...,
        interface_index: int = ...,
    ) -> None: ...
    @overload
    def add_tcp_reader(
        self, port: int, max_size: int = ..., buffer_size: int = ..., bind_hostname: str = ...
    ) -> None: ...
    @overload
    def add_tcp_reader(self, acceptor: socket.socket, max_size: int = ...) -> None: ...
    def add_udp_ibv_reader(self, config: UdpIbvConfig) -> None: ...
    def add_udp_pcap_file_reader(self, filename: str, filter: str = ...) -> None: ...
    def add_inproc_reader(self, queue: spead2.InprocQueue) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def stats(self) -> StreamStats: ...
    @property
    def config(self) -> StreamConfig: ...

# We make a dummy _RingStream base class because mypy objects to the async stream
# type overloading get as a coroutine.
class _RingStream(_Stream):
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        config: StreamConfig = ...,
        ring_config: RingStreamConfig = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator[Heap]: ...
    def get_nowait(self) -> Heap: ...
    @property
    def fd(self) -> int: ...
    @property
    def ringbuffer(self) -> Ringbuffer: ...
    @property
    def ring_config(self) -> RingStreamConfig: ...

class Stream(_RingStream):
    def get(self) -> Heap: ...

class ChunkStreamConfig:
    DEFAULT_MAX_CHUNKS: ClassVar[int]

    items: list[int]
    max_chunks: int
    place: tuple | None
    max_heap_extra: int
    def enable_packet_presence(self, payload_size: int) -> None: ...
    def disable_packet_presence(self) -> None: ...
    @property
    def packet_presence_payload_size(self) -> int: ...
    def __init__(
        self,
        *,
        items: list[int] = ...,
        max_chunks: int = ...,
        place: tuple | None = ...,
        max_heap_extra: int = ...,
    ) -> None: ...

class Chunk:
    chunk_id: int
    stream_id: int
    present: object  # optional buffer protocol
    data: object  # optional buffer protocol
    extra: object  # optional buffer protocol

    def __init__(
        self,
        *,
        chunk_id: int = ...,
        stream_id: int = ...,
        present: object = ...,
        data: object = ...,
        extra: object = ...,
    ) -> None: ...

# Dummy base class because the async ChunkRingbuffer.get has a different
# signature to the synchronous version.
class _ChunkRingbuffer:
    def __init__(self, maxsize: int) -> None: ...
    def qsize(self) -> int: ...
    @property
    def maxsize(self) -> int: ...
    @property
    def data_fd(self) -> int: ...
    @property
    def free_fd(self) -> int: ...
    def get_nowait(self) -> Chunk: ...
    def put_nowait(self, chunk: Chunk) -> None: ...
    def empty(self) -> bool: ...
    def full(self) -> bool: ...
    def stop(self) -> bool: ...
    def add_producer(self) -> None: ...
    def remove_producer(self) -> bool: ...

class ChunkRingbuffer(_ChunkRingbuffer):
    def get(self) -> Chunk: ...
    def put(self, chunk: Chunk) -> None: ...
    def __iter__(self) -> Iterator[Chunk]: ...

class ChunkRingPair:
    def add_free_chunk(self, chunk: Chunk) -> None: ...
    @property
    def data_ringbuffer(self) -> _ChunkRingbuffer: ...
    @property
    def free_ringbuffer(self) -> _ChunkRingbuffer: ...

class ChunkRingStream(_Stream, ChunkRingPair):
    def __init__(
        self,
        thread_pool: spead2.ThreadPool,
        config: StreamConfig,
        chunk_stream_config: ChunkStreamConfig,
        data_ringbuffer: _ChunkRingbuffer,
        free_ringbuffer: _ChunkRingbuffer,
    ) -> None: ...

class ChunkStreamGroupConfig:
    class EvictionMode(enum.Enum):
        LOSSY = ...
        LOSSLESS = ...
    DEFAULT_MAX_CHUNKS: ClassVar[int]
    @property
    def max_chunks(self) -> int: ...
    @property
    def eviction_mode(self) -> ChunkStreamGroupConfig.EvictionMode: ...
    def __init__(self, *, max_chunks=..., eviction_mode=...) -> None: ...

class ChunkStreamRingGroup(ChunkRingPair, collections.abc.Sequence[ChunkStreamGroupMember]):
    def __init__(
        self,
        config: ChunkStreamGroupConfig,
        data_ringbuffer: _ChunkRingbuffer,
        free_ringbuffer: _ChunkRingbuffer,
    ) -> None: ...
    @property
    def config(self) -> ChunkStreamGroupConfig: ...
    def emplace_back(
        self,
        thread_pool: spead2.ThreadPool,
        config: spead2.recv.StreamConfig,
        chunk_stream_config: spead2.recv.ChunkStreamConfig,
    ) -> ChunkStreamGroupMember: ...
    def stop(self) -> None: ...
    # These are marked abstract in Sequence, so need to be implemented here
    @overload
    def __getitem__(self, index: int) -> ChunkStreamGroupMember: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[ChunkStreamGroupMember]: ...
    def __len__(self) -> int: ...

class ChunkStreamGroupMember(_Stream): ...
