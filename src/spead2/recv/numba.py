# Copyright 2021-2022 National Research Foundation (SARAO)
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

"""Utilities for writing chunking receive callbacks using Numba."""

from numba import types

try:
    _size_t = types.size_t
except AttributeError:
    # Older versions of numba.types doesn't have a size_t, so assume it is the same as uintptr_t
    _size_t = types.uintp

chunk_place_data = types.Record.make_c_struct(
    [
        ("packet", types.intp),  # uint8_t *
        ("packet_size", _size_t),
        ("items", types.intp),  # s_item_pointer_t *
        ("chunk_id", types.int64),
        ("heap_index", _size_t),
        ("heap_offset", _size_t),
        ("batch_stats", types.intp),  # uint64_t *
        ("extra", types.intp),  # uint8_t *
        ("extra_offset", _size_t),
        ("extra_size", _size_t),
    ]
)
"""Numba record type representing the C structure used in the chunk placement callback.

Numba doesn't (as of 0.54) support pointers in records, so the pointer fields
are represented as integers. Use :py:func:`spead2.numba.intp_to_voidptr` to
convert them to void pointers then :py:func:`numba.carray` to convert the void
pointer to an array of the appropriate size and dtype.
"""
