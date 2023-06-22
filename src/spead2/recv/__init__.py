# Copyright 2015, 2020-2021 National Research Foundation (SARAO)
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

"""Receive SPEAD protocol

Item format
===========
At present only a subset of the possible SPEAD format strings are accepted.
Also, the SPEAD protocol does not specify how items are to be represented in
Python. The following are accepted.

 - Any descriptor with a numpy header (will be handled by numpy). If the dtype
   contains only a single field which is non-native endian, it will be
   converted to native endian in-place. In other cases, the value retrieved
   from numpy will still be correct, but usage may be slow.
 - If no numpy header is present, the following may be used in the format
   with zero copy and good efficiency:

   - u8, u16, u32, u64
   - i8, i16, i32, i64
   - f32, f64
   - b8
   - c8 (converted to dtype S1)

   This will be converted to a numpy dtype. If there are multiple fields,
   their names will be generated by numpy (`f0`, `f1`, etc). At most one
   element of the shape may indicate a variable-length field, whose length
   will be computed from the size of the item, or 0 if any other element of
   the shape is zero.
 - The `u`, `i`, `c` and `b` types may also be used with other sizes, but it
   will invoke a slow conversion process and is not recommended for large
   arrays. For `c`, the value is interpreted as a Unicode code point.

Two cases are treated specially:

 - A zero-dimensional array is returned as a scalar, rather than a
   zero-dimensional array object.
 - A one-dimensional array of characters (numpy dtype 'S1') is converted to a
   Python string, using ASCII encoding.

Immediate values are treated as items with heap_address_bits/8
bytes, in the order they appeared in the original packet.
"""

from spead2._spead2.recv import (           # noqa: F401
    Chunk,
    ChunkRingStream,
    ChunkRingbuffer,
    ChunkStreamConfig,
    ChunkStreamGroupConfig,
    ChunkStreamGroupMember,
    ChunkStreamRingGroup,
    Heap,
    IncompleteHeap,
    RingStreamConfig,
    Stream,
    StreamConfig,
    StreamStatConfig,
    StreamStats,
)
from . import stream_stat_indices           # noqa: F401
try:
    from spead2._spead2.recv import UdpIbvConfig      # noqa: F401
except ImportError:
    pass
