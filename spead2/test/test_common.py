"""Tests for parts of spead2 that are shared between send and receive"""

from __future__ import division, print_function
import spead2
import numpy as np
from nose.tools import *

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
