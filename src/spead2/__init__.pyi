# Copyright 2019-2021 National Research Foundation (SARAO)
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

from typing import (List, Sequence, Optional, Tuple, Any, Union,
                    Dict, KeysView, ValuesView, Text, overload)

import numpy as np
try:
    from numpy.typing import DTypeLike as _DTypeLike
except ImportError:
    _DTypeLike = Any  # type: ignore
import spead2.recv

_EndpointList = List[Tuple[str, int]]

__version__: str

BUG_COMPAT_DESCRIPTOR_WIDTHS: int
BUG_COMPAT_SHAPE_BIT_1: int
BUG_COMPAT_SWAP_ENDIAN: int
BUG_COMPAT_PYSPEAD_0_5_2: int

NULL_ID: int
HEAP_CNT_ID: int
HEAP_LENGTH_ID: int
PAYLOAD_OFFSET_ID: int
PAYLOAD_LENGTH_ID: int
DESCRIPTOR_ID: int
STREAM_CTRL_ID: int

DESCRIPTOR_NAME_ID: int
DESCRIPTOR_DESCRIPTION_ID: int
DESCRIPTOR_SHAPE_ID: int
DESCRIPTOR_FORMAT_ID: int
DESCTIPTOR_DTYPE_ID: int

CTRL_STREAM_START: int
CTRL_DESCRIPTOR_REISSUE: int
CTRL_STREAM_STOP: int
CTRL_DESCRIPTOR_UPDATE: int

MEMCPY_STD: int
MEMCPY_NONTEMPORAL: int
MEMCPY_NONTEMPORAL_RW: int

class Stopped(RuntimeError):
    pass

class Empty(RuntimeError):
    pass

class Flavour:
    @overload
    def __init__(self, version: int, item_pointer_bits : int,
                 heap_address_bits: int, bug_compat: int = ...) -> None: ...
    @overload
    def __init__(self) -> None: ...
    def __eq__(self, o: object) -> bool: ...
    def __ne__(self, o: object) -> bool: ...
    @property
    def version(self) -> int: ...
    @property
    def item_pointer_bits(self) -> int: ...
    @property
    def heap_address_bits(self) -> int: ...
    @property
    def bug_compat(self) -> int: ...

class ThreadPool:
    @overload
    def __init__(self, threads: int = ...) -> None: ...
    @overload
    def __init__(self, threads: int, affinity: List[int]) -> None: ...

class MemoryAllocator:
    def __init__(self) -> None: ...

class MmapAllocator(MemoryAllocator):
    def __init__(self, flags: int = ...) -> None: ...

class MemoryPool(MemoryAllocator):
    warn_on_empty: bool

    @overload
    def __init__(self, lower: int, upper: int, max_free: int, initial: int,
                 allocator: Optional[MemoryAllocator] = None) -> None: ...
    @overload
    def __init__(self, thread_pool: ThreadPool, lower: int, upper: int, max_free: int, initial: int,
                 low_water: int, allocator: MemoryAllocator) -> None: ...


class InprocQueue:
    def __init__(self) -> None: ...
    def add_packet(self, packet) -> None: ...
    def stop(self) -> None: ...

class RawDescriptor:
    id: int
    name: bytes
    description: bytes
    numpy_header: bytes
    format: List[Tuple[str, int]]

    @property
    def shape(self) -> List[Optional[int]]: ...
    @shape.setter
    def shape(self, value: Sequence[Optional[int]]) -> None: ...

class IbvContext:
    def __init__(self, interface_address: str) -> None: ...
    def reset(self) -> None: ...

def parse_range_list(ranges: str) -> List[int]: ...

class Descriptor:
    id: int
    name: str
    description: str
    shape: Sequence[Optional[int]]
    dtype: Optional[np.dtype]
    order: str
    format: Optional[List[Tuple[str, int]]]

    def __init__(self, id: int, name: str, description: str,
                 shape: Sequence[Optional[int]], dtype: Optional[_DTypeLike] = None,
                 order: str = ..., format: Optional[List[Tuple[str, int]]] = None) -> None: ...
    @property
    def itemsize_bits(self) -> int: ...
    @property
    def is_variable_size(self) -> bool: ...
    @property
    def allow_immediate(self) -> bool: ...
    def dynamic_shape(self, max_elements: int) -> Sequence[int]: ...
    def compatible_shape(self, shape: Sequence[int]) -> bool: ...
    @classmethod
    def from_raw(cls, raw_descriptor: RawDescriptor, flavour: Flavour): ...
    def to_raw(self, flavour: Flavour) -> RawDescriptor: ...

class Item(Descriptor):
    version: int

    def __init__(self, id: int, name: str, description: str,
                 shape: Sequence[Optional[int]], dtype: Optional[_DTypeLike] = None,
                 order: str = ..., format: Optional[List[Tuple[str, int]]] = None,
                 value: Any = None) -> None: ...
    @property
    def value(self) -> Any: ...
    @value.setter
    def value(self, new_value: Any) -> None: ...
    def set_from_raw(self, raw_item: spead2.recv.RawItem, new_order: str = ...) -> None: ...
    # typing has no buffer protocol ABC (https://bugs.python.org/issue27501)
    def to_buffer(self) -> Any: ...

class ItemGroup:
    def __init__(self) -> None: ...
    def add_item(self, id: Optional[int], name: str, description: str,
                 shape: Sequence[Optional[int]], dtype: Optional[_DTypeLike] = None,
                 order: str = 'C', format: Optional[List[Tuple[str, int]]] = None,
                 value: Any = None) -> Item: ...
    def __getitem__(self, key: Union[int, str]) -> Item: ...
    def __contains__(self, key: Union[int, str]) -> bool: ...
    def keys(self) -> KeysView[str]: ...
    def ids(self) -> KeysView[int]: ...
    def values(self) -> ValuesView[Item]: ...
    def __len__(self) -> int: ...
    def update(self, heap: spead2.recv.Heap, new_order: str = ...) -> Dict[str, Item]: ...
