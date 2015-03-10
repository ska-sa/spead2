from __future__ import division, print_function, unicode_literals
import spead2
import spead2.send as send
import struct
import binascii
import numpy as np
from nose.tools import *
from .defines import *

def hexlify(packets):
    return [binascii.hexlify(x) for x in packets]


class Flavour(object):
    def __init__(self, heap_address_bits, bug_compat=0):
        self.heap_address_bits = heap_address_bits
        self.bug_compat = bug_compat

    def make_header(self, num_items):
        address_size = self.heap_address_bits // 8
        item_size = 8 - address_size
        return struct.pack('>Q', 0x5304000000000000 | (address_size << 32) | (item_size << 40) | num_items)

    def make_immediate(self, item_id, value):
        return struct.pack('>Q', 2**63 | (item_id << self.heap_address_bits) | value)

    def make_address(self, item_id, address):
        return struct.pack('>Q', (item_id << self.heap_address_bits) | address)

    def _encode_be(self, size, value):
        """Encodes `value` as big-endian in `size` bytes"""
        assert size <= 8
        packed = struct.pack('>Q', value)
        return packed[8 - size:]

    def make_shape(self, shape):
        # TODO: extend for bug_compat flavours
        ans = []
        for size in shape:
            if size < 0:
                ans.append(struct.pack('B', 1))
                ans.append(self._encode_be(self.heap_address_bits // 8, 0))
            else:
                ans.append(struct.pack('B', 0))
                ans.append(self._encode_be(self.heap_address_bits // 8, size))
        return b''.join(ans)

    def items_to_bytes(self, items, descriptors=None, max_packet_size=1500):
        if descriptors is None:
            descriptors = items
        heap = send.Heap(0x123456, 0)
        for descriptor in descriptors:
            heap.add_descriptor(descriptor)
        for item in items:
            heap.add_item(item)
        gen = send.PacketGenerator(heap, self.heap_address_bits, self.bug_compat, max_packet_size)
        return list(gen)


def offset_generator(fields):
    offset = 0
    yield offset
    for field in fields:
        offset += len(field)
        yield offset


class TestEncode(object):
    """Test heap encoding of various data"""

    def __init__(self):
        self.flavour = Flavour(48, 0)

    def test_empty(self):
        """An empty heap must still generate a packet"""
        expected = [
            b''.join([
                self.flavour.make_header(4),
                self.flavour.make_immediate(HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(HEAP_LENGTH_ID, 8),
                self.flavour.make_immediate(PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(PAYLOAD_LENGTH_ID, 8),
                struct.pack('>Q', 0)])
        ]
        packet = self.flavour.items_to_bytes([])
        assert_equal(hexlify(expected), hexlify(packet))

    def make_descriptor_numpy(self, id, name, description, shape, dtype_str, fortran_order):
        payload_fields = [
            b'name',
            b'description',
            b'',
            self.flavour.make_shape(shape),
            b"{{'descr': {!r}, 'fortran_order': {!r}, 'shape': {!r}}}".format(
                dtype_str, bool(fortran_order), tuple(shape))
        ]
        payload = b''.join(payload_fields)
        offsets = offset_generator(payload_fields)
        descriptor = b''.join([
            self.flavour.make_header(10),
            self.flavour.make_immediate(HEAP_CNT_ID, 1),
            self.flavour.make_immediate(HEAP_LENGTH_ID, len(payload)),
            self.flavour.make_immediate(PAYLOAD_OFFSET_ID, 0),
            self.flavour.make_immediate(PAYLOAD_LENGTH_ID, len(payload)),
            self.flavour.make_immediate(DESCRIPTOR_ID_ID, id),
            self.flavour.make_address(DESCRIPTOR_NAME_ID, next(offsets)),
            self.flavour.make_address(DESCRIPTOR_DESCRIPTION_ID, next(offsets)),
            self.flavour.make_address(DESCRIPTOR_FORMAT_ID, next(offsets)),
            self.flavour.make_address(DESCRIPTOR_SHAPE_ID, next(offsets)),
            self.flavour.make_address(DESCRIPTOR_DTYPE_ID, next(offsets)),
            payload
        ])
        return descriptor

    def test_simple_numpy(self):
        id = 0x2345
        shape = (2, 3)
        data = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.uint16)
        payload_fields = [
            self.make_descriptor_numpy(id, b'name', b'description', shape, b'<u2', False),
            bytes(data.data)
        ]
        payload = b''.join(payload_fields)
        offsets = offset_generator(payload_fields)
        expected = [
            b''.join([
                self.flavour.make_header(6),
                self.flavour.make_immediate(HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(HEAP_LENGTH_ID, len(payload)),
                self.flavour.make_immediate(PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(PAYLOAD_LENGTH_ID, len(payload)),
                self.flavour.make_address(DESCRIPTOR_ID, next(offsets)),
                self.flavour.make_address(id, next(offsets)),
                payload
            ])
        ]

        item = spead2.Item(id=id, name=b'name', description=b'description', shape=shape, dtype=np.uint16)
        item.value = data
        packet = self.flavour.items_to_bytes([item])
        assert_equal(hexlify(expected), hexlify(packet))
