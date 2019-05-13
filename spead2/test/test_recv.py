# Copyright 2015, 2017 SKA South Africa
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

from __future__ import division, print_function
import socket
import struct
import time

import numpy as np
import six
from nose.tools import (
    assert_equal, assert_in, assert_is_instance,
    assert_true, assert_false, assert_raises)
try:
    from nose.tools import assert_logs    # Only available from 3.4
except ImportError:
    assert_logs = None

import spead2
import spead2.recv as recv
import spead2.send as send

from .test_common import assert_equal_typed


class Item(object):
    def __init__(self, id, value, immediate=False, offset=0):
        self.id = id
        self.value = value
        self.immediate = immediate
        self.offset = offset

    def encode(self, heap_address_bits):
        if self.immediate:
            return struct.pack('>Q', (1 << 63) | (self.id << heap_address_bits) | self.value)
        else:
            return struct.pack('>Q', (self.id << heap_address_bits) | self.offset)


class Flavour(object):
    def __init__(self, heap_address_bits, bug_compat=0):
        self.heap_address_bits = heap_address_bits
        self.bug_compat = bug_compat

    def _encode_be(self, size, value):
        """Encodes `value` as big-endian in `size` bytes"""
        assert size <= 8
        packed = struct.pack('>Q', value)
        return packed[8 - size:]

    def make_packet(self, items, payload):
        """Generate data for a packet at a low level. The value of non-immediate
        items are taken from the payload; the values in the objects are ignored.
        """
        data = []
        data.append(struct.pack('>BBBBHH', 0x53, 0x4,
                                (64 - self.heap_address_bits) // 8,
                                self.heap_address_bits // 8, 0,
                                len(items)))
        for item in items:
            data.append(item.encode(self.heap_address_bits))
        data.append(bytes(payload))
        return b''.join(data)

    def make_format(self, format):
        if self.bug_compat & spead2.BUG_COMPAT_DESCRIPTOR_WIDTHS:
            field_size = 7
        else:
            field_size = (64 - self.heap_address_bits) // 8
        data = []
        for field in format:
            assert len(field[0]) == 1
            data.append(field[0].encode('ascii'))
            data.append(self._encode_be(field_size, field[1]))
        return b''.join(data)

    def make_shape(self, shape):
        if self.bug_compat & spead2.BUG_COMPAT_DESCRIPTOR_WIDTHS:
            field_size = 8
        else:
            field_size = self.heap_address_bits // 8 + 1
        if self.bug_compat & spead2.BUG_COMPAT_SHAPE_BIT_1:
            variable_marker = 2
        else:
            variable_marker = 1

        data = []
        for value in shape:
            if value is None:
                data.append(struct.pack('>B', variable_marker))
                data.append(self._encode_be(field_size - 1, 0))
            elif value >= 0:
                data.append(self._encode_be(field_size, value))
            else:
                raise ValueError('Shape must contain non-negative values and None')
        return b''.join(data)

    def make_packet_heap(self, heap_cnt, items):
        """Construct a single-packet heap."""
        payload_size = 0
        for item in items:
            if not item.immediate:
                payload_size += len(bytes(item.value))

        all_items = [
            Item(spead2.HEAP_CNT_ID, heap_cnt, True),
            Item(spead2.PAYLOAD_OFFSET_ID, 0, True),
            Item(spead2.PAYLOAD_LENGTH_ID, payload_size, True),
            Item(spead2.HEAP_LENGTH_ID, payload_size, True)]
        offset = 0
        payload = bytearray(payload_size)
        for item in items:
            if not item.immediate:
                value = bytes(item.value)
                all_items.append(Item(item.id, value, offset=offset))
                payload[offset : offset + len(value)] = value
                offset += len(value)
            else:
                all_items.append(item)
        return self.make_packet(all_items, payload)

    def make_plain_descriptor(self, id, name, description, format, shape):
        if not isinstance(name, bytes):
            name = name.encode('ascii')
        if not isinstance(description, bytes):
            description = description.encode('ascii')
        return Item(spead2.DESCRIPTOR_ID, self.make_packet_heap(
            1,
            [
                Item(spead2.DESCRIPTOR_ID_ID, id, True),
                Item(spead2.DESCRIPTOR_NAME_ID, name),
                Item(spead2.DESCRIPTOR_DESCRIPTION_ID, description),
                Item(spead2.DESCRIPTOR_FORMAT_ID, self.make_format(format)),
                Item(spead2.DESCRIPTOR_SHAPE_ID, self.make_shape(shape))
            ]))

    def make_numpy_descriptor_raw(self, id, name, description, header):
        if not isinstance(name, bytes):
            name = name.encode('ascii')
        if not isinstance(description, bytes):
            description = description.encode('ascii')
        if not isinstance(header, bytes):
            header = header.encode('ascii')
        return Item(spead2.DESCRIPTOR_ID, self.make_packet_heap(
            1,
            [
                Item(spead2.DESCRIPTOR_ID_ID, id, True),
                Item(spead2.DESCRIPTOR_NAME_ID, name),
                Item(spead2.DESCRIPTOR_DESCRIPTION_ID, description),
                Item(spead2.DESCRIPTOR_DTYPE_ID, header)
            ]))

    def make_numpy_descriptor(self, id, name, description, dtype, shape, fortran_order=False):
        header = str({
            'descr': np.lib.format.dtype_to_descr(np.dtype(dtype)),
            'fortran_order': bool(fortran_order),
            'shape': tuple(shape)
        })
        return self.make_numpy_descriptor_raw(id, name, description, header)

    def make_numpy_descriptor_from(self, id, name, description, array):
        if array.flags.c_contiguous:
            fortran_order = False
        elif array.flags.f_contiguous:
            fortran_order = True
        else:
            raise ValueError('Array must be C or Fortran-order contiguous')
        return self.make_numpy_descriptor(
            id, name, description, array.dtype, array.shape, fortran_order)


FLAVOUR = Flavour(48)


class TestDecode(object):
    """Various types of descriptors must be correctly interpreted to decode data"""

    def __init__(self):
        self.flavour = FLAVOUR

    def data_to_heaps(self, data, **kwargs):
        """Take some data and pass it through the receiver to obtain a set of heaps.

        Keyword arguments are passed to the receiver constructor.
        """
        thread_pool = spead2.ThreadPool()
        stop_on_stop_item = kwargs.pop('stop_on_stop_item', None)
        allow_unsized_heaps = kwargs.pop('allow_unsized_heaps', None)
        stream = recv.Stream(thread_pool, self.flavour.bug_compat, **kwargs)
        if stop_on_stop_item is not None:
            stream.stop_on_stop_item = stop_on_stop_item
        if allow_unsized_heaps is not None:
            stream.allow_unsized_heaps = allow_unsized_heaps
        stream.add_buffer_reader(data)
        return list(stream)

    def data_to_ig(self, data):
        """Take some data and pass it through the receiver to obtain a single heap,
        from which the items are extracted.
        """
        heaps = self.data_to_heaps(data)
        assert_equal(1, len(heaps))
        ig = spead2.ItemGroup()
        ig.update(heaps[0])
        for name, item in ig.items():
            assert_equal_typed(name, item.name)
        return ig

    def data_to_item(self, data, expected_id):
        """Take some data and pass it through the receiver to obtain a single heap,
        with a single item, which is returned.
        """
        ig = self.data_to_ig(data)
        assert_equal(1, len(ig))
        assert_in(expected_id, ig)
        return ig[expected_id]

    def test_scalar_int(self):
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_scalar_int', 'a scalar integer', [('i', 32)], []),
                Item(0x1234, struct.pack('>i', -123456789))
            ])
        item = self.data_to_item(packet, 0x1234)
        assert_is_instance(item.value, np.int32)
        assert_equal(-123456789, item.value)

    def test_scalar_int_immediate(self):
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_scalar_int', 'a scalar integer', [('u', 32)], []),
                Item(0x1234, 0x12345678, True)
            ])
        item = self.data_to_item(packet, 0x1234)
        assert isinstance(item.value, np.uint32)
        assert_equal(0x12345678, item.value)

    def test_scalar_int_immediate_fastpath(self):
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_scalar_uint', 'a scalar integer', [('u', 48)], []),
                self.flavour.make_plain_descriptor(
                    0x1235, 'test_scalar_positive', 'a positive scalar integer', [('i', 48)], []),
                self.flavour.make_plain_descriptor(
                    0x1236, 'test_scalar_negative', 'a negative scalar integer', [('i', 48)], []),
                Item(0x1234, 0x9234567890AB, True),
                Item(0x1235, 0x1234567890AB, True),
                Item(0x1236, 0x9234567890AB, True)
            ])
        ig = self.data_to_ig(packet)
        assert_equal(3, len(ig))
        assert_equal(0x9234567890AB, ig[0x1234].value)
        assert_equal(0x1234567890AB, ig[0x1235].value)
        assert_equal(-0x6DCBA9876F55, ig[0x1236].value)

    def test_string(self):
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_string', 'a byte string', [('c', 8)], [None]),
                Item(0x1234, 'Hello world'.encode('ascii'))
            ])
        item = self.data_to_item(packet, 0x1234)
        assert_equal_typed('Hello world', item.value)

    def test_array(self):
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_array', 'an array of floats', [('f', 32)], (3, 2)),
                Item(0x1234, struct.pack('>6f', *np.arange(1.5, 7.5)))
            ])
        item = self.data_to_item(packet, 0x1234)
        expected = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=np.float32)
        np.testing.assert_equal(expected, item.value)

    def test_array_fields(self):
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_array', 'an array of floats', [('f', 32), ('i', 8)], (3,)),
                Item(0x1234, struct.pack('>fbfbfb', 1.5, 1, 2.5, 2, 4.5, -4))
            ])
        item = self.data_to_item(packet, 0x1234)
        dtype = np.dtype('=f4,i1')
        assert_equal(dtype, item.value.dtype)
        expected = np.array([(1.5, 1), (2.5, 2), (4.5, -4)], dtype=dtype)
        np.testing.assert_equal(expected, item.value)

    def test_array_numpy(self):
        expected = np.array([[1.25, 1.75], [2.25, 2.5]], dtype=np.float64)
        send = expected
        if self.flavour.bug_compat & spead2.BUG_COMPAT_SWAP_ENDIAN:
            send = expected.byteswap()
        else:
            send = expected
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_numpy_descriptor_from(
                    0x1234, 'test_array_numpy', 'an array of floats', expected),
                Item(0x1234, send.data)
            ])
        item = self.data_to_item(packet, 0x1234)
        assert_equal(expected.dtype, item.value.dtype)
        np.testing.assert_equal(expected, item.value)

    def test_array_numpy_fortran(self):
        expected = np.array([[1.25, 1.75], [2.25, 2.5]], dtype=np.float64).T
        assert_false(expected.flags.c_contiguous)
        send = expected
        if self.flavour.bug_compat & spead2.BUG_COMPAT_SWAP_ENDIAN:
            send = expected.byteswap()
        else:
            send = expected
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_numpy_descriptor_from(
                    0x1234, 'test_array_numpy_fortran', 'an array of floats', expected),
                Item(0x1234, np.ravel(send, 'K').data)
            ])
        item = self.data_to_item(packet, 0x1234)
        assert_equal(expected.dtype, item.value.dtype)
        np.testing.assert_equal(expected, item.value)

    def test_fallback_uint(self):
        expected = np.array([0xABC, 0xDEF, 0x123])
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_fallback_uint', 'an array of 12-bit uints', [('u', 12)], (3,)),
                Item(0x1234, b'\xAB\xCD\xEF\x12\x30')
            ])
        item = self.data_to_item(packet, 0x1234)
        np.testing.assert_equal(expected, item.value)

    def test_fallback_int(self):
        expected = np.array([-1348, -529, 291])
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_fallback_uint', 'an array of 12-bit ints', [('i', 12)], (3,)),
                Item(0x1234, b'\xAB\xCD\xEF\x12\x30')
            ])
        item = self.data_to_item(packet, 0x1234)
        np.testing.assert_equal(expected, item.value)

    def test_fallback_types(self):
        expected = np.array([(True, 17, 'y', 1.0), (False, -23, 'n', -1.0)], dtype='O,O,S1,>f4')
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_fallback_uint', 'an array with bools, int, chars and floats',
                    [('b', 1), ('i', 7), ('c', 8), ('f', 32)], (2,)),
                Item(0x1234, b'\x91y\x3F\x80\x00\x00' + b'\x69n\xBF\x80\x00\x00')
            ])
        item = self.data_to_item(packet, 0x1234)
        np.testing.assert_equal(expected, item.value)

    def test_fallback_scalar(self):
        expected = 0x1234567890AB
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_fallback_scalar', 'a scalar with unusual type',
                    [('u', 48)], ()),
                Item(0x1234, b'\x12\x34\x56\x78\x90\xAB')
            ])
        item = self.data_to_item(packet, 0x1234)
        assert_equal(expected, item.value)

    def test_duplicates(self):
        payload = bytearray(64)
        payload[:] = range(64)
        packets = [
            self.flavour.make_packet([
                Item(spead2.HEAP_CNT_ID, 1, True),
                Item(spead2.PAYLOAD_OFFSET_ID, 0, True),
                Item(spead2.PAYLOAD_LENGTH_ID, 32, True),
                Item(spead2.HEAP_LENGTH_ID, 64, True),
                Item(0x1600, 12345, True),
                Item(0x5000, 0, False, offset=0)], payload[0 : 32]),
            self.flavour.make_packet([
                Item(spead2.HEAP_CNT_ID, 1, True),
                Item(spead2.PAYLOAD_OFFSET_ID, 32, True),
                Item(spead2.PAYLOAD_LENGTH_ID, 32, True),
                Item(spead2.HEAP_LENGTH_ID, 64, True),
                Item(0x1600, 12345, True),
                Item(0x5000, 0, False, offset=0)], payload[32 : 64])
        ]
        heaps = self.data_to_heaps(b''.join(packets))
        assert_equal(1, len(heaps))
        items = heaps[0].get_items()
        assert_equal(2, len(items))
        items.sort(key=lambda item: item.id)
        assert_equal(0x1600, items[0].id)
        assert_true(items[0].is_immediate)
        assert_equal(12345, items[0].immediate_value)
        assert_equal(0x5000, items[1].id)
        assert_false(items[1].is_immediate)
        assert_equal(payload, bytearray(items[1]))

    def test_incomplete_heaps(self):
        payload = bytearray(64)
        payload[:] = range(64)
        packets = [
            self.flavour.make_packet([
                Item(spead2.HEAP_CNT_ID, 1, True),
                Item(spead2.PAYLOAD_OFFSET_ID, 5, True),
                Item(spead2.PAYLOAD_LENGTH_ID, 7, True),
                Item(spead2.HEAP_LENGTH_ID, 96, True),
                Item(0x1600, 12345, True),
                Item(0x5000, 0, False, offset=0)], payload[5 : 12]),
            self.flavour.make_packet([
                Item(spead2.HEAP_CNT_ID, 1, True),
                Item(spead2.PAYLOAD_OFFSET_ID, 32, True),
                Item(spead2.PAYLOAD_LENGTH_ID, 32, True),
                Item(spead2.HEAP_LENGTH_ID, 96, True)], payload[32 : 64])
        ]
        heaps = self.data_to_heaps(b''.join(packets),
                                   contiguous_only=False,
                                   incomplete_keep_payload_ranges=True)
        assert_equal(1, len(heaps))
        assert_is_instance(heaps[0], recv.IncompleteHeap)
        items = heaps[0].get_items()
        assert_equal(1, len(items))   # Addressed item must be excluded
        assert_equal(0x1600, items[0].id)
        assert_equal(96, heaps[0].heap_length)
        assert_equal(39, heaps[0].received_length)
        assert_equal([(5, 12), (32, 64)], heaps[0].payload_ranges)

    def test_is_start_of_stream(self):
        packet = self.flavour.make_packet_heap(
            1,
            [Item(spead2.STREAM_CTRL_ID, spead2.CTRL_STREAM_START, immediate=True)])
        heaps = self.data_to_heaps(packet)
        assert_true(heaps[0].is_start_of_stream())

        packet = self.flavour.make_packet_heap(
            1,
            [Item(spead2.STREAM_CTRL_ID, spead2.CTRL_DESCRIPTOR_REISSUE, immediate=True)])
        heaps = self.data_to_heaps(packet)
        assert_false(heaps[0].is_start_of_stream())

    def test_is_end_of_stream(self):
        packet = self.flavour.make_packet_heap(
            1,
            [Item(spead2.STREAM_CTRL_ID, spead2.CTRL_STREAM_STOP, immediate=True)])
        heaps = self.data_to_heaps(packet, stop_on_stop_item=False)
        assert_true(heaps[0].is_end_of_stream())
        assert_false(heaps[0].is_start_of_stream())

    def test_no_stop_on_stop_item(self):
        packet1 = self.flavour.make_packet_heap(
            1,
            [Item(spead2.STREAM_CTRL_ID, spead2.CTRL_STREAM_STOP, immediate=True)])
        packet2 = self.flavour.make_packet_heap(
            2,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_string', 'a byte string', [('c', 8)], [None]),
                Item(0x1234, 'Hello world'.encode('ascii'))
            ])
        heaps = self.data_to_heaps(packet1 + packet2, stop_on_stop_item=False)
        assert_equal(2, len(heaps))
        ig = spead2.ItemGroup()
        ig.update(heaps[0])
        ig.update(heaps[1])
        assert_equal_typed('Hello world', ig['test_string'].value)

    def test_size_mismatch(self):
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'bad', 'an item with insufficient data', [('u', 32)], (5, 5)),
                Item(0x1234, b'\0' * 99)
            ])
        heaps = self.data_to_heaps(packet)
        assert_equal(1, len(heaps))
        ig = spead2.ItemGroup()
        assert_raises(ValueError, ig.update, heaps[0])

    def test_numpy_object(self):
        """numpy dtypes can contain Python objects (by pointer). These can't be
        used for SPEAD.
        """
        dtype = np.dtype('f4,O')
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_numpy_descriptor(
                    0x1234, 'object', 'an item with object pointers', dtype, (5,)),
                Item(0x1234, b'?' * 100)
            ])
        heaps = self.data_to_heaps(packet)
        assert_equal(1, len(heaps))
        ig = spead2.ItemGroup()
        assert_raises(ValueError, ig.update, heaps[0])

    def test_numpy_zero_size(self):
        """numpy dtypes can represent zero bytes."""
        dtype = np.dtype(np.str_)
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_numpy_descriptor(
                    0x1234, 'empty', 'an item with zero-byte dtype', dtype, (5,)),
                Item(0x1234, b'')
            ])
        heaps = self.data_to_heaps(packet)
        assert_equal(1, len(heaps))
        ig = spead2.ItemGroup()
        assert_raises(ValueError, ig.update, heaps[0])

    def test_numpy_malformed(self):
        """Malformed numpy header must raise :py:exc:`ValueError`."""
        def helper(header):
            packet = self.flavour.make_packet_heap(
                1,
                [
                    self.flavour.make_numpy_descriptor_raw(
                        0x1234, 'name', 'description', header)
                ])
            heaps = self.data_to_heaps(packet)
            assert_equal(1, len(heaps))
            ig = spead2.ItemGroup()
            assert_raises(ValueError, ig.update, heaps[0])
        helper("{'descr': 'S1'")   # Syntax error: no closing brace
        helper("123")              # Not a dictionary
        helper("import os")        # Security check
        helper("{'descr': 'S1'}")  # Missing keys
        helper("{'descr': 'S1', 'fortran_order': False, 'shape': (), 'foo': 'bar'}")  # Extra keys
        helper("{'descr': 'S1', 'fortran_order': False, 'shape': (-1,)}")    # Bad shape
        helper("{'descr': 1, 'fortran_order': False, 'shape': ()}")          # Bad descriptor
        helper("{'descr': '+-', 'fortran_order': False, 'shape': ()}")       # Bad descriptor
        helper("{'descr': 'S1', 'fortran_order': 0, 'shape': ()}")           # Bad fortran_order
        helper("{'descr': 'S1', 'fortran_order': False, 'shape': (None,)}")  # Bad shape

    def test_nonascii_value(self):
        """Receiving non-ASCII characters in a c8 string must raise
        :py:exc:`UnicodeDecodeError`."""
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'test_string', 'a byte string', [('c', 8)], [None]),
                Item(0x1234, six.u('\u0200').encode('utf-8'))
            ])
        heaps = self.data_to_heaps(packet)
        ig = spead2.ItemGroup()
        assert_raises(UnicodeDecodeError, ig.update, heaps[0])

    def test_nonascii_name(self):
        """Receiving non-ASCII characters in an item name must raise
        :py:exc:`UnicodeDecodeError`."""
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, b'\xEF', 'a byte string', [('c', 8)], [None])
            ])
        heaps = self.data_to_heaps(packet)
        ig = spead2.ItemGroup()
        assert_raises(UnicodeDecodeError, ig.update, heaps[0])

    def test_nonascii_description(self):
        """Receiving non-ASCII characters in an item description must raise
        :py:exc:`UnicodeDecodeError`."""
        packet = self.flavour.make_packet_heap(
            1,
            [
                self.flavour.make_plain_descriptor(
                    0x1234, 'name', b'\xEF', [('c', 8)], [None])
            ])
        heaps = self.data_to_heaps(packet)
        ig = spead2.ItemGroup()
        assert_raises(UnicodeDecodeError, ig.update, heaps[0])

    def test_no_heap_size(self):
        """Heap consisting of packets with no heap size items must work.
        This also tests mixing the general items in with the packet items.
        """
        payload1 = bytes(np.arange(0, 64, dtype=np.uint8).data)
        payload2 = bytes(np.arange(64, 96, dtype=np.uint8).data)
        packet1 = self.flavour.make_packet(
            [
                Item(spead2.HEAP_CNT_ID, 1, True),
                Item(0x1000, None, False, offset=0),
                Item(spead2.PAYLOAD_OFFSET_ID, 0, True),
                Item(spead2.PAYLOAD_LENGTH_ID, 64, True)
            ], payload1)
        packet2 = self.flavour.make_packet(
            [
                Item(spead2.HEAP_CNT_ID, 1, True),
                Item(spead2.PAYLOAD_OFFSET_ID, 64, True),
                Item(spead2.PAYLOAD_LENGTH_ID, 32, True)
            ], payload2)
        heaps = self.data_to_heaps(packet1 + packet2)
        assert_equal(1, len(heaps))
        raw_items = heaps[0].get_items()
        assert_equal(1, len(raw_items))
        assert_equal(payload1 + payload2, bytearray(raw_items[0]))

    def test_disallow_unsized_heaps(self):
        """Packets without heap length rejected if disallowed"""
        packet = self.flavour.make_packet(
            [
                Item(spead2.HEAP_CNT_ID, 1, True),
                Item(0x1000, None, False, offset=0),
                Item(spead2.PAYLOAD_OFFSET_ID, 0, True),
                Item(spead2.PAYLOAD_LENGTH_ID, 64, True)
            ], bytes(np.arange(0, 64, dtype=np.uint8).data))
        if assert_logs is not None:
            with assert_logs('spead2', 'INFO') as cm:
                heaps = self.data_to_heaps(packet, allow_unsized_heaps=False)
                # Logging is asynchronous, so we have to give it a bit of time
                time.sleep(0.1)
            assert_equal(cm.output, ['INFO:spead2:packet rejected because it has no HEAP_LEN'])
        else:
            # Python 2 fallback
            heaps = self.data_to_heaps(packet, allow_unsized_heaps=False)
        assert_equal(0, len(heaps))

    def test_bad_offset(self):
        """Heap with out-of-range offset should be dropped"""
        packet = self.flavour.make_packet(
            [
                Item(spead2.HEAP_CNT_ID, 1, True),
                Item(spead2.HEAP_LENGTH_ID, 64, True),
                Item(spead2.PAYLOAD_OFFSET_ID, 0, True),
                Item(spead2.PAYLOAD_LENGTH_ID, 64, True),
                Item(0x1000, None, False, offset=65)
            ], b'\0' * 64)
        heaps = self.data_to_heaps(packet)
        assert_equal(0, len(heaps))


class TestStream(object):
    """Tests for the stream API"""

    def __init__(self):
        self.flavour = FLAVOUR

    def test_max_heaps_zero(self):
        """Constructing a stream with max_heaps=0 raises ValueError"""
        with assert_raises(ValueError):
            spead2.recv.Stream(spead2.ThreadPool(), max_heaps=0)

    def test_full_stop(self):
        """Must be able to stop even if the consumer is not consuming
        anything."""
        thread_pool = spead2.ThreadPool(1)
        sender = send.BytesStream(thread_pool)
        ig = send.ItemGroup()
        data = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.uint16)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=data.shape, dtype=data.dtype, value=data)
        gen = send.HeapGenerator(ig)
        for i in range(10):
            sender.send_heap(gen.get_heap(data='all'))
        receiver = spead2.recv.Stream(thread_pool, ring_heaps=4)
        receiver.add_buffer_reader(sender.getvalue())
        # Wait for the ring buffer to block
        while receiver.stats.worker_blocked == 0:
            time.sleep(0.0)
        # Can't usefully check the stats here, because they're only
        # updated at the end of a batch.
        assert_equal(4, receiver.ringbuffer.capacity())
        assert_equal(4, receiver.ringbuffer.size())

        receiver.stop()            # This unblocks all remaining heaps
        stats = receiver.stats
        assert_equal(10, stats.heaps)
        assert_equal(10, stats.packets)
        assert_equal(0, stats.incomplete_heaps_evicted)
        assert_equal(0, stats.incomplete_heaps_flushed)
        assert_equal(1, stats.worker_blocked)
        assert_equal(4, receiver.ringbuffer.capacity())
        assert_equal(4, receiver.ringbuffer.size())

    def test_no_stop_heap(self):
        """A heap containing a stop is not passed to the ring"""
        thread_pool = spead2.ThreadPool(1)
        sender = send.BytesStream(thread_pool)
        ig = send.ItemGroup()
        data = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.uint16)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=data.shape, dtype=data.dtype, value=data)
        gen = send.HeapGenerator(ig)
        sender.send_heap(gen.get_heap(data='all'))
        sender.send_heap(gen.get_end())
        receiver = spead2.recv.Stream(thread_pool)
        receiver.add_buffer_reader(sender.getvalue())
        heaps = list(receiver)
        assert_equal(1, len(heaps))

        stats = receiver.stats
        assert_equal(1, stats.heaps)
        assert_equal(2, stats.packets)
        assert_equal(0, stats.incomplete_heaps_evicted)
        assert_equal(0, stats.incomplete_heaps_flushed)
        assert_equal(0, stats.worker_blocked)


class TestUdpReader(object):
    def test_out_of_range_udp_port(self):
        receiver = spead2.recv.Stream(spead2.ThreadPool())
        assert_raises(TypeError, receiver.add_udp_reader, 100000)

    def test_illegal_udp_port(self):
        receiver = spead2.recv.Stream(spead2.ThreadPool())
        assert_raises(RuntimeError, receiver.add_udp_reader, 22)


class TestTcpReader(object):
    def setup(self):
        self.receiver = spead2.recv.Stream(spead2.ThreadPool())
        recv_sock = socket.socket()
        recv_sock.bind(("127.0.0.1", 0))
        recv_sock.listen(1)
        port = recv_sock.getsockname()[1]
        self.receiver.add_tcp_reader(acceptor=recv_sock)
        recv_sock.close()

        self.send_sock = socket.socket()
        self.send_sock.connect(("127.0.0.1", port))

    def teardown(self):
        self.close()

    def close(self):
        if self.send_sock is not None:
            self.send_sock.close()
            self.send_sock = None

    def data_to_heaps(self, data):
        """Take some data and pass it through the receiver to obtain a set of heaps.

        The send socket is closed.
        """
        self.send_sock.send(data)
        self.close()
        return list(self.receiver)

    def data_to_ig(self, data):
        """Take some data and pass it through the receiver to obtain a single heap,
        from which the items are extracted.
        """
        heaps = self.data_to_heaps(data)
        assert_equal(1, len(heaps))
        ig = spead2.ItemGroup()
        ig.update(heaps[0])
        for name, item in ig.items():
            assert_equal_typed(name, item.name)
        return ig

    def data_to_item(self, data, expected_id):
        """Take some data and pass it through the receiver to obtain a single heap,
        with a single item, which is returned.
        """
        ig = self.data_to_ig(data)
        assert_equal(1, len(ig))
        assert_in(expected_id, ig)
        return ig[expected_id]

    def simple_packet(self):
        return FLAVOUR.make_packet_heap(
            1,
            [
                FLAVOUR.make_plain_descriptor(
                    0x1234, 'test_scalar_int', 'a scalar integer', [('u', 32)], []),
                Item(0x1234, 0x12345678, True)
            ])

    def test_bad_header(self):
        """A nonsense header followed by a normal packet"""
        data = b'deadbeef' + self.simple_packet()
        item = self.data_to_item(data, 0x1234)
        assert_is_instance(item.value, np.uint32)
        assert_equal(0x12345678, item.value)

    def test_packet_too_big(self):
        """A packet that is too large (should be rejected) followed by a normal packet"""
        zeros = np.zeros(100000, np.uint8)
        packet1 = FLAVOUR.make_packet_heap(
            1,
            [
                FLAVOUR.make_numpy_descriptor_from(
                    0x2345, 'test_big_array', 'over-sized packet', zeros),
                Item(0x2345, zeros.data)
            ])
        data = packet1 + self.simple_packet()
        item = self.data_to_item(data, 0x1234)
        assert_is_instance(item.value, np.uint32)
        assert_equal(0x12345678, item.value)

    def test_partial_header(self):
        """Connection closed partway through a header"""
        packet = self.simple_packet()
        heaps = self.data_to_heaps(packet[:6])
        assert_equal([], heaps)

    def test_partial_packet(self):
        """Connection closed partway through item descriptors"""
        packet = self.simple_packet()
        heaps = self.data_to_heaps(packet[:45])
        assert_equal([], heaps)

    def test_partial_payload(self):
        """Connection closed partway through item payload"""
        packet = self.simple_packet()
        heaps = self.data_to_heaps(packet[:-1])
        assert_equal([], heaps)
