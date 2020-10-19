# Copyright 2015, 2019-2020 National Research Foundation (SARAO)
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

"""Tests for parts of spead2 that are shared between send and receive"""

import numpy as np
import pytest

import spead2


class TestParseRangeList:
    def test_empty(self):
        assert spead2.parse_range_list('') == []

    def test_simple(self):
        assert spead2.parse_range_list('1,2,5') == [1, 2, 5]

    def test_ranges(self):
        assert spead2.parse_range_list('100,4-6,8,10-10,12-13') == [100, 4, 5, 6, 8, 10, 12, 13]


class TestThreadPool:
    """Smoke tests for :py:class:`spead2.ThreadPool`. These are very simple
    tests, because it is not actually possible to check things like the
    thread affinity."""
    def test_simple(self):
        spead2.ThreadPool()
        spead2.ThreadPool(4)

    def test_affinity(self):
        spead2.ThreadPool(3, [])
        spead2.ThreadPool(3, [0, 1])
        spead2.ThreadPool(1, [1, 0, 2])

    def test_zero_threads(self):
        with pytest.raises(ValueError):
            spead2.ThreadPool(0)
        with pytest.raises(ValueError):
            spead2.ThreadPool(0, [0, 1])


class TestFlavour:
    def test_bad_version(self):
        with pytest.raises(ValueError):
            spead2.Flavour(3, 64, 40, 0)

    def test_bad_item_pointer_bits(self):
        with pytest.raises(ValueError):
            spead2.Flavour(4, 32, 24, 0)

    def test_bad_heap_address_bits(self):
        with pytest.raises(ValueError):
            spead2.Flavour(4, 64, 43, 0)  # Not multiple of 8
        with pytest.raises(ValueError):
            spead2.Flavour(4, 64, 0, 0)
        with pytest.raises(ValueError):
            spead2.Flavour(4, 64, 64, 0)

    def test_attributes(self):
        flavour = spead2.Flavour(4, 64, 40, 2)
        assert flavour.version == 4
        assert flavour.item_pointer_bits == 64
        assert flavour.heap_address_bits == 40
        assert flavour.bug_compat == 2

    def test_equality(self):
        flavour1 = spead2.Flavour(4, 64, 40, 2)
        flavour1b = spead2.Flavour(4, 64, 40, 2)
        flavour2 = spead2.Flavour(4, 64, 48, 2)
        flavour3 = spead2.Flavour(4, 64, 40, 4)
        assert flavour1 == flavour1b
        assert flavour1 != flavour2
        assert not (flavour1 != flavour1b)
        assert not (flavour1 == flavour2)
        assert not (flavour1 == flavour3)


class TestItem:
    """Tests for :py:class:`spead2.Item`.

    Many of these actually test :py:class:`spead2.Descriptor`, but since the
    internals of these classes are interwined, it is simpler to keep all the
    tests together here."""

    def test_nonascii_value(self):
        """Using a non-ASCII unicode character raises a
        :py:exc:`UnicodeEncodeError`."""
        item1 = spead2.Item(0x1000, 'name1', 'description',
                            (None,), format=[('c', 8)], value='\u0200')
        item2 = spead2.Item(0x1001, 'name2', 'description2', (),
                            dtype='S5', value='\u0201')
        with pytest.raises(UnicodeEncodeError):
            item1.to_buffer()
        with pytest.raises(UnicodeEncodeError):
            item2.to_buffer()

    def test_format_and_dtype(self):
        """Specifying both a format and dtype raises :py:exc:`ValueError`."""
        with pytest.raises(ValueError):
            spead2.Item(0x1000, 'name', 'description',
                        (1, 2), format=[('c', 8)], dtype='S1')

    def test_no_format_or_dtype(self):
        """At least one of format and dtype must be specified."""
        with pytest.raises(ValueError):
            spead2.Item(0x1000, 'name', 'description', (1, 2), format=None)

    def test_invalid_order(self):
        """The `order` parameter must be either 'C' or 'F'."""
        with pytest.raises(ValueError):
            spead2.Item(0x1000, 'name', 'description', (1, 2), np.int32, order='K')

    def test_fortran_fallback(self):
        """The `order` parameter must be 'C' for legacy formats."""
        with pytest.raises(ValueError):
            spead2.Item(0x1000, 'name', 'description', (1, 2), format=[('u', 32)], order='F')

    def test_empty_format(self):
        """Format must not be empty"""
        with pytest.raises(ValueError):
            spead2.Item(0x1000, 'name', 'description', (1, 2), format=[])

    def test_assign_none(self):
        """Changing a value back to `None` raises :py:exc:`ValueError`."""
        item = spead2.Item(0x1000, 'name', 'description', (), np.int32)
        with pytest.raises(ValueError):
            item.value = None

    def test_multiple_unknown(self):
        """Multiple unknown dimensions are not allowed."""
        with pytest.raises(ValueError):
            spead2.Item(0x1000, 'name', 'description', (5, None, 3, None), format=[('u', 32)])

    def test_numpy_unknown(self):
        """Unknown dimensions are not permitted when using a numpy descriptor"""
        with pytest.raises(ValueError):
            spead2.Item(0x1000, 'name', 'description', (5, None), np.int32)

    def test_nonascii_name(self):
        """Name with non-ASCII characters must fail"""
        with pytest.raises(UnicodeEncodeError):
            item = spead2.Item(0x1000, '\u0200', 'description', (), np.int32)
            item.to_raw(spead2.Flavour())

    def test_nonascii_description(self):
        """Description with non-ASCII characters must fail"""
        with pytest.raises(UnicodeEncodeError):
            item = spead2.Item(0x1000, 'name', '\u0200', (), np.int32)
            item.to_raw(spead2.Flavour())


class TestItemGroup:
    """Tests for :py:class:`spead2.ItemGroup`"""

    def test_allocate_id(self):
        """Automatic allocation of IDs must skip over already-allocated IDs"""
        ig = spead2.ItemGroup()
        ig.add_item(0x1000, 'item 1', 'item 1', (), np.int32)
        ig.add_item(0x1003, 'item 2', 'item 2', (), np.int32)
        ig.add_item(None, 'item 3', 'item 3', (), np.int32)
        ig.add_item(None, 'item 4', 'item 4', (), np.int32)
        ig.add_item(None, 'item 5', 'item 5', (), np.int32)
        assert ig[0x1001].name == 'item 3'
        assert ig[0x1002].name == 'item 4'
        assert ig[0x1004].name == 'item 5'

    def test_replace_rename(self):
        """When a new item is added with a known ID but a different name, the
        old item must cease to exist."""
        ig = spead2.ItemGroup()
        ig.add_item(0x1000, 'item 1', 'item 1', (), np.int32)
        ig.add_item(0x1000, 'renamed', 'renamed', (), np.int32)
        assert list(ig.ids()) == [0x1000]
        assert list(ig.keys()) == ['renamed']

    def test_replace_clobber_name(self):
        """When a new item is added that collides with an existing name, that
        other item is deleted."""
        ig = spead2.ItemGroup()
        ig.add_item(0x1000, 'item 1', 'item 1', (), np.int32)
        ig.add_item(0x1001, 'item 1', 'clobbered', (), np.int32)
        assert list(ig.ids()) == [0x1001]
        assert list(ig.keys()) == ['item 1']
        assert ig['item 1'].description == 'clobbered'
        assert ig['item 1'] is ig[0x1001]

    def test_replace_reset_value(self):
        """When a descriptor is replaced with an identical one but a new
        value, the new value must take effect and the version must be
        incremented."""
        ig = spead2.ItemGroup()
        ig.add_item(0x1000, 'item 1', 'item 1', (), np.int32, value=np.int32(4))
        ig.add_item(0x1001, 'item 2', 'item 2', (), np.int32, value=np.int32(5))
        item1 = ig[0x1000]
        item2 = ig[0x1001]
        item1_version = item1.version
        item2_version = item2.version
        ig.add_item(0x1000, 'item 1', 'item 1', (), np.int32, value=np.int32(6))
        ig.add_item(0x1001, 'item 2', 'item 2', (), np.int32)
        assert item1 is ig[0x1000]
        assert item2 is ig[0x1001]
        assert item1.value == np.int32(6)
        assert item1.version > item1_version
        assert item2.value == np.int32(5)
        assert item2_version == item2.version

    def test_replace_change_shape(self):
        """When a descriptor changes the shape, the old item must be discarded
        and ``None`` used in its place. The version must be bumped."""
        ig = spead2.ItemGroup()
        ig.add_item(0x1000, 'item 1', 'item 1', (), np.int32, value=np.int32(4))
        item1 = ig[0x1000]
        item1_version = item1.version
        ig.add_item(0x1000, 'item 1', 'bigger', (3, 4), np.int32)
        assert item1 is not ig[0x1000]
        assert ig[0x1000].value is None
        assert ig[0x1000].version > item1_version

    def test_replace_clobber_both(self):
        """Adding a new item that collides with one item on name and another on
        ID causes both to be dropped."""
        ig = spead2.ItemGroup()
        ig.add_item(0x1000, 'item 1', 'item 1', (), np.int32)
        ig.add_item(0x1001, 'item 2', 'item 2', (), np.int32)
        ig.add_item(0x1000, 'item 2', 'clobber', (), np.int32)
        assert list(ig.ids()) == [0x1000]
        assert list(ig.keys()) == ['item 2']
        assert ig[0x1000] is ig['item 2']
        assert ig[0x1000].description == 'clobber'
