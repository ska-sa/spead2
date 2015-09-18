# Copyright 2015 SKA South Africa
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

from __future__ import division, print_function
import spead2
import numpy as np
from nose.tools import *


def assert_equal_typed(expected, actual, msg=None):
    """Check that expected and actual compare equal *and* have the same type.

    This is used for checking that strings have the correct type (str vs
    unicode in Python 2, str vs bytes in Python 3).
    """
    assert_equal(expected, actual, msg)
    assert_equal(type(expected), type(actual), msg)


class TestFlavour(object):
    def test_bad_version(self):
        with assert_raises(ValueError):
            spead2.Flavour(3, 64, 40, 0)

    def test_bad_item_pointer_bits(self):
        with assert_raises(ValueError):
            spead2.Flavour(4, 32, 24, 0)

    def test_bad_heap_address_bits(self):
        with assert_raises(ValueError):
            spead2.Flavour(4, 64, 43, 0)  # Not multiple of 8
        with assert_raises(ValueError):
            spead2.Flavour(4, 64, 0, 0)
        with assert_raises(ValueError):
            spead2.Flavour(4, 64, 64, 0)

    def test_attributes(self):
        flavour = spead2.Flavour(4, 64, 40, 2)
        assert_equal(4, flavour.version)
        assert_equal(64, flavour.item_pointer_bits)
        assert_equal(40, flavour.heap_address_bits)
        assert_equal(2, flavour.bug_compat)

    def test_equality(self):
        flavour1 = spead2.Flavour(4, 64, 40, 2)
        flavour1b = spead2.Flavour(4, 64, 40, 2)
        flavour2 = spead2.Flavour(4, 64, 48, 2)
        flavour3 = spead2.Flavour(4, 64, 40, 4)
        assert_true(flavour1 == flavour1b)
        assert_true(flavour1 != flavour2)
        assert_false(flavour1 != flavour1b)
        assert_false(flavour1 == flavour2)
        assert_false(flavour1 == flavour3)


class TestItem(object):
    """Tests for :py:class:`spead2.Item`"""

    def test_nonascii_value(self):
        """Using a non-ASCII unicode character raises a
        :py:exc:`UnicodeEncodeError`."""
        item1 = spead2.Item(0x1000, 'name1', 'description',
            (None,), format=[('c', 8)], value=u'\u0200')
        item2 = spead2.Item(0x1001, 'name2', 'description2', (),
            dtype='S5', value=u'\u0201')
        assert_raises(UnicodeEncodeError, item1.to_buffer)
        assert_raises(UnicodeEncodeError, item2.to_buffer)

    def test_format_and_dtype(self):
        """Specifying both a format and dtype raises :py:exc:`ValueError`."""
        assert_raises(ValueError, spead2.Item, 0x1000, 'name', 'description',
            (1, 2), format=[('c', 8)], dtype='S1')

    def test_no_format_or_dtype(self):
        """At least one of format and dtype must be specified."""
        assert_raises(ValueError, spead2.Item, 0x1000, 'name', 'description',
            (1, 2), format=None)

    def test_invalid_order(self):
        """The `order` parameter must be either 'C' or 'F'."""
        assert_raises(ValueError, spead2.Item, 0x1000, 'name', 'description',
            (1, 2), np.int32, order='K')

    def test_fortran_fallback(self):
        """The `order` parameter must be either 'C' for legacy formats."""
        assert_raises(ValueError, spead2.Item, 0x1000, 'name', 'description',
            (1, 2), format=[('u', 32)], order='F')

    def test_empty_format(self):
        """Format must not be empty"""
        assert_raises(ValueError, spead2.Item, 0x1000, 'name', 'description',
            (1, 2), format=[])

    def test_assign_none(self):
        """Changing a value back to `None` raises :py:exc:`ValueError`."""
        item = spead2.Item(0x1000, 'name', 'description', (), np.int32)
        with assert_raises(ValueError):
            item.value = None

    def test_multiple_unknown(self):
        """Multiple unknown dimensions are not allowed."""
        assert_raises(ValueError, spead2.Item, 0x1000, 'name', 'description',
            (5, None, 3, None), format=[('u', 32)])

    def test_numpy_unknown(self):
        """Unknown dimensions are not permitted when using a numpy descriptor"""
        assert_raises(ValueError, spead2.Item, 0x1000, 'name', 'description',
            (5, None), np.int32)


class TestItemGroup(object):
    """Tests for :py:class:`spead2.ItemGroup`"""

    def test_allocate_id(self):
        ig = spead2.ItemGroup()
        ig.add_item(0x1000, 'item 1', 'item 1', (), np.int32)
        ig.add_item(0x1003, 'item 2', 'item 2', (), np.int32)
        ig.add_item(None, 'item 3', 'item 3', (), np.int32)
        ig.add_item(None, 'item 4', 'item 4', (), np.int32)
        ig.add_item(None, 'item 5', 'item 5', (), np.int32)
        assert_equal('item 3', ig[0x1001].name)
        assert_equal('item 4', ig[0x1002].name)
        assert_equal('item 5', ig[0x1004].name)
