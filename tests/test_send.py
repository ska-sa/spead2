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

import binascii
import gc
import math
import struct
import time
import threading
import weakref

import numpy as np
import pytest

import spead2
import spead2.send as send


def hexlify(data):
    """Turns a byte string into human-readable hex dump"""
    if isinstance(data, list):
        return [hexlify(x) for x in data]
    chunks = []
    for i in range(0, len(data), 8):
        part = data[i : min(i + 8, len(data))]
        chunks.append(b':'.join([binascii.hexlify(part[i : i + 1]) for i in range(len(part))]))
    return b'\n'.join(chunks).decode('ascii')


def encode_be(size, value):
    """Encodes `value` as big-endian in `size` bytes"""
    assert size <= 8
    packed = struct.pack('>Q', value)
    return packed[8 - size:]


class Flavour(spead2.Flavour):
    def __init__(self, version, item_pointer_bits, heap_address_bits, bug_compat=0):
        super().__init__(version, item_pointer_bits, heap_address_bits, bug_compat)

    def make_header(self, num_items):
        address_size = self.heap_address_bits // 8
        item_size = 8 - address_size
        return struct.pack(
            '>Q',
            0x5304000000000000 | (address_size << 32) | (item_size << 40) | num_items)

    def make_immediate(self, item_id, value):
        return struct.pack('>Q', 2**63 | (item_id << self.heap_address_bits) | value)

    def make_address(self, item_id, address):
        return struct.pack('>Q', (item_id << self.heap_address_bits) | address)

    def make_shape(self, shape):
        # TODO: extend for bug_compat flavours
        assert not (self.bug_compat &
                    (spead2.BUG_COMPAT_DESCRIPTOR_WIDTHS | spead2.BUG_COMPAT_SHAPE_BIT_1))
        ans = []
        for size in shape:
            if size < 0:
                ans.append(struct.pack('B', 1))
                ans.append(encode_be(self.heap_address_bits // 8, 0))
            else:
                ans.append(struct.pack('B', 0))
                ans.append(encode_be(self.heap_address_bits // 8, size))
        return b''.join(ans)

    def make_format(self, format):
        # TODO: extend for bug_compat flavours
        assert not self.bug_compat & spead2.BUG_COMPAT_DESCRIPTOR_WIDTHS
        ans = []
        for (code, length) in format:
            ans.append(struct.pack('B', ord(code)))
            ans.append(encode_be(8 - self.heap_address_bits // 8, length))
        return b''.join(ans)

    def items_to_bytes(self, items, descriptors=None, max_packet_size=1500, repeat_pointers=False):
        if descriptors is None:
            descriptors = items
        heap = send.Heap(self)
        for descriptor in descriptors:
            heap.add_descriptor(descriptor)
        for item in items:
            heap.add_item(item)
        heap.repeat_pointers = repeat_pointers
        gen = send.PacketGenerator(heap, 0x123456, max_packet_size)
        return list(gen)


def offset_generator(fields):
    offset = 0
    yield offset
    for field in fields:
        offset += len(field)
        yield offset


class TestEncode:
    """Test heap encoding of various data"""

    def setup(self):
        self.flavour = Flavour(4, 64, 48, 0)

    def test_empty(self):
        """An empty heap must still generate a packet"""
        expected = [
            b''.join([
                self.flavour.make_header(5),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 1),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 1),
                self.flavour.make_address(spead2.NULL_ID, 0),
                struct.pack('B', 0)])
        ]
        packet = self.flavour.items_to_bytes([])
        assert hexlify(packet) == hexlify(expected)

    def test_lifetime(self):
        """Heap must hold references to item values"""
        item = spead2.Item(id=0x2345, name='name', description='description',
                           shape=(2, 3), dtype=np.uint16)
        item.value = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.uint16)
        weak = weakref.ref(item.value)
        heap = send.Heap(self.flavour)
        heap.add_item(item)
        del item
        packets = list(send.PacketGenerator(heap, 0x123456, 1472))   # noqa: F841
        assert weak() is not None
        del heap
        # pypy needs multiple gc passes to wind it all up
        for i in range(10):
            gc.collect()
        assert weak() is None

    def make_descriptor_numpy(self, id, name, description, shape, dtype_str, fortran_order):
        payload_fields = [
            b'name',
            b'description',
            b'',
            self.flavour.make_shape(shape),
            "{{'descr': {!r}, 'fortran_order': {!r}, 'shape': {!r}}}".format(
                str(dtype_str), bool(fortran_order), tuple(shape)).encode()
        ]
        payload = b''.join(payload_fields)
        offsets = offset_generator(payload_fields)
        descriptor = b''.join([
            self.flavour.make_header(10),
            self.flavour.make_immediate(spead2.HEAP_CNT_ID, 1),
            self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, len(payload)),
            self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
            self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, len(payload)),
            self.flavour.make_immediate(spead2.DESCRIPTOR_ID_ID, id),
            self.flavour.make_address(spead2.DESCRIPTOR_NAME_ID, next(offsets)),
            self.flavour.make_address(spead2.DESCRIPTOR_DESCRIPTION_ID, next(offsets)),
            self.flavour.make_address(spead2.DESCRIPTOR_FORMAT_ID, next(offsets)),
            self.flavour.make_address(spead2.DESCRIPTOR_SHAPE_ID, next(offsets)),
            self.flavour.make_address(spead2.DESCRIPTOR_DTYPE_ID, next(offsets)),
            payload
        ])
        return descriptor

    def make_descriptor_fallback(self, id, name, description, shape, format):
        payload_fields = [
            b'name',
            b'description',
            self.flavour.make_format(format),
            self.flavour.make_shape(shape),
        ]
        payload = b''.join(payload_fields)
        offsets = offset_generator(payload_fields)
        descriptor = b''.join([
            self.flavour.make_header(9),
            self.flavour.make_immediate(spead2.HEAP_CNT_ID, 1),
            self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, len(payload)),
            self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
            self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, len(payload)),
            self.flavour.make_immediate(spead2.DESCRIPTOR_ID_ID, id),
            self.flavour.make_address(spead2.DESCRIPTOR_NAME_ID, next(offsets)),
            self.flavour.make_address(spead2.DESCRIPTOR_DESCRIPTION_ID, next(offsets)),
            self.flavour.make_address(spead2.DESCRIPTOR_FORMAT_ID, next(offsets)),
            self.flavour.make_address(spead2.DESCRIPTOR_SHAPE_ID, next(offsets)),
            payload
        ])
        return descriptor

    def test_numpy_simple(self):
        """A single numpy-format item with descriptor"""
        id = 0x2345
        shape = (2, 3)
        data = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.uint16)
        payload_fields = [
            self.make_descriptor_numpy(id, 'name', 'description', shape, '<u2', False),
            struct.pack('<6H', 6, 7, 8, 10, 11, 12000)
        ]
        payload = b''.join(payload_fields)
        offsets = offset_generator(payload_fields)
        expected = [
            b''.join([
                self.flavour.make_header(6),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, len(payload)),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, len(payload)),
                self.flavour.make_address(spead2.DESCRIPTOR_ID, next(offsets)),
                self.flavour.make_address(id, next(offsets)),
                payload
            ])
        ]

        item = spead2.Item(id=id, name='name', description='description',
                           shape=shape, dtype=np.uint16)
        item.value = data
        packet = self.flavour.items_to_bytes([item])
        assert hexlify(packet) == hexlify(expected)

    def test_numpy_noncontiguous(self):
        """A numpy item with a discontiguous item value is sent correctly"""
        id = 0x2345
        shape = (2, 3)
        store = np.array([[6, 7, 8, 0, 1], [10, 11, 12000, 2, 3], [9, 9, 9, 9, 9]], dtype=np.uint16)
        data = store[:2, :3]
        payload = struct.pack('<6H', 6, 7, 8, 10, 11, 12000)
        expected = [
            b''.join([
                self.flavour.make_header(5),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, len(payload)),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, len(payload)),
                self.flavour.make_address(id, 0),
                payload
            ])
        ]
        item = spead2.Item(id=id, name='name', description='description',
                           shape=shape, dtype=np.uint16)
        item.value = data
        packet = self.flavour.items_to_bytes([item], [])
        assert hexlify(packet) == hexlify(expected)

    def test_numpy_fortran_order(self):
        """A numpy item with Fortran-order descriptor must be sent in Fortran order"""
        id = 0x2345
        shape = (2, 3)
        data = np.array([[6, 7, 8], [10, 11, 12000]], order='F')
        assert data.flags.c_contiguous is False
        payload_fields = [
            self.make_descriptor_numpy(id, 'name', 'description', shape, '<u2', True),
            struct.pack('<6H', 6, 10, 7, 11, 8, 12000)
        ]
        payload = b''.join(payload_fields)
        offsets = offset_generator(payload_fields)
        expected = [
            b''.join([
                self.flavour.make_header(6),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, len(payload)),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, len(payload)),
                self.flavour.make_address(spead2.DESCRIPTOR_ID, next(offsets)),
                self.flavour.make_address(id, next(offsets)),
                payload
            ])
        ]
        item = spead2.Item(id=id, name='name', description='description',
                           shape=shape, dtype=np.uint16, order='F')
        item.value = data
        packet = self.flavour.items_to_bytes([item])
        assert hexlify(packet) == hexlify(expected)

    def test_fallback_types(self):
        """Send an array with mixed types and strange packing"""
        id = 0x2345
        format = [('b', 1), ('i', 7), ('c', 8), ('f', 32)]
        shape = (2,)
        data = [(True, 17, 'y', 1.0), (False, -23.0, 'n', -1.0)]
        payload_fields = [
            self.make_descriptor_fallback(id, 'name', 'description', shape, format),
            b'\x91y\x3F\x80\x00\x00' + b'\x69n\xBF\x80\x00\x00'
        ]
        payload = b''.join(payload_fields)
        offsets = offset_generator(payload_fields)
        expected = [
            b''.join([
                self.flavour.make_header(6),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, len(payload)),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, len(payload)),
                self.flavour.make_address(spead2.DESCRIPTOR_ID, next(offsets)),
                self.flavour.make_address(id, next(offsets)),
                payload
            ])
        ]
        item = spead2.Item(id=id, name='name', description='description',
                           shape=shape, format=format)
        item.value = data
        packet = self.flavour.items_to_bytes([item])
        assert hexlify(packet) == hexlify(expected)

    def test_small_fixed(self):
        """Sending a small item with fixed shape must use an immediate."""
        id = 0x2345
        data = 0x7654
        expected = [
            b''.join([
                self.flavour.make_header(6),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 1),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 1),
                self.flavour.make_immediate(id, data),
                self.flavour.make_address(spead2.NULL_ID, 0),
                struct.pack('B', 0)
            ])
        ]
        item = spead2.Item(id=id, name='name', description='description',
                           shape=(), format=[('u', 16)])
        item.value = data
        packet = self.flavour.items_to_bytes([item], [])
        assert hexlify(packet) == hexlify(expected)

    def test_small_variable(self):
        """Sending a small item with dynamic shape must not use an immediate."""
        id = 0x2345
        shape = (1, None)
        data = np.array([[4, 5]], dtype=np.uint8)
        payload = struct.pack('>2B', 4, 5)
        expected = [
            b''.join([
                self.flavour.make_header(5),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, len(payload)),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, len(payload)),
                self.flavour.make_address(id, 0),
                payload
            ])
        ]
        item = spead2.Item(id=id, name='name', description='description',
                           shape=shape, format=[('u', 8)])
        item.value = data
        packet = self.flavour.items_to_bytes([item], [])
        assert hexlify(packet) == hexlify(expected)

    def test_numpy_zero_length(self):
        """A zero-length numpy type raises :exc:`ValueError`"""
        with pytest.raises(ValueError):
            spead2.Item(id=0x2345, name='name', description='description',
                        shape=(), dtype=np.str_)

    def test_fallback_zero_length(self):
        """A zero-length type raises :exc:`ValueError`"""
        with pytest.raises(ValueError):
            spead2.Item(id=0x2345, name='name', description='description',
                        shape=(), format=[('u', 0)])

    def test_start(self):
        """Tests sending a start-of-stream marker."""
        expected = [
            b''.join([
                self.flavour.make_header(6),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 1),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 1),
                self.flavour.make_immediate(spead2.STREAM_CTRL_ID, spead2.CTRL_STREAM_START),
                self.flavour.make_address(spead2.NULL_ID, 0),
                struct.pack('B', 0)
            ])
        ]
        heap = send.Heap(self.flavour)
        heap.add_start()
        packet = list(send.PacketGenerator(heap, 0x123456, 1500))
        assert hexlify(packet) == hexlify(expected)

    def test_end(self):
        """Tests sending an end-of-stream marker."""
        expected = [
            b''.join([
                self.flavour.make_header(6),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 1),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 1),
                self.flavour.make_immediate(spead2.STREAM_CTRL_ID, spead2.CTRL_STREAM_STOP),
                self.flavour.make_address(spead2.NULL_ID, 0),
                struct.pack('B', 0)
            ])
        ]
        heap = send.Heap(self.flavour)
        heap.add_end()
        packet = list(send.PacketGenerator(heap, 0x123456, 1500))
        assert hexlify(packet) == hexlify(expected)

    def test_replicate_pointers(self):
        """Tests sending a heap with replicate_pointers set to true"""
        id = 0x2345
        data = np.arange(32, dtype=np.uint8)
        item1 = spead2.Item(id=id, name='item1', description='addressed item',
                            shape=data.shape, dtype=data.dtype, value=data)
        item2 = spead2.Item(id=id + 1, name='item2', description='inline item',
                            shape=(), format=[('u', self.flavour.heap_address_bits)],
                            value=0xdeadbeef)
        expected = [
            b''.join([
                self.flavour.make_header(6),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 32),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 16),
                self.flavour.make_address(id, 0),
                self.flavour.make_immediate(id + 1, 0xdeadbeef),
                data.tobytes()[0:16]
            ]),
            b''.join([
                self.flavour.make_header(6),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, 0x123456),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 32),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 16),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 16),
                self.flavour.make_address(id, 0),
                self.flavour.make_immediate(id + 1, 0xdeadbeef),
                data.tobytes()[16:32]
            ])
        ]
        packets = self.flavour.items_to_bytes([item1, item2], [], max_packet_size=72,
                                              repeat_pointers=True)
        assert hexlify(packets) == hexlify(expected)


class TestStreamConfig:
    def test_default_construct(self):
        config = send.StreamConfig()
        assert config.max_packet_size == config.DEFAULT_MAX_PACKET_SIZE
        assert config.rate == 0.0
        assert config.burst_size == config.DEFAULT_BURST_SIZE
        assert config.max_heaps == config.DEFAULT_MAX_HEAPS
        assert config.burst_rate_ratio == config.DEFAULT_BURST_RATE_RATIO
        assert config.rate_method == config.DEFAULT_RATE_METHOD

    def test_setters(self):
        config = send.StreamConfig()
        config.max_packet_size = 1234
        config.rate = 1e9
        config.burst_size = 12345
        config.max_heaps = 5
        config.burst_rate_ratio = 1.5
        config.rate_method = send.RateMethod.SW
        assert config.max_packet_size == 1234
        assert config.rate == 1e9
        assert config.burst_size == 12345
        assert config.max_heaps == 5
        assert config.burst_rate_ratio == 1.5
        assert config.rate_method == send.RateMethod.SW
        assert config.burst_rate == 1.5e9

    def test_construct_kwargs(self):
        config = send.StreamConfig(max_packet_size=1234, max_heaps=5)
        assert config.max_packet_size == 1234
        assert config.max_heaps == 5
        assert config.rate == 0.0

    def test_bad_rate(self):
        with pytest.raises(ValueError):
            send.StreamConfig(rate=-1.0)
        with pytest.raises(ValueError):
            send.StreamConfig(rate=math.nan)

    def test_bad_max_heaps(self):
        with pytest.raises(TypeError):
            send.StreamConfig(max_heaps=-1)
        with pytest.raises(ValueError):
            send.StreamConfig(max_heaps=0)


class TestStream:
    def setup(self):
        # A slow stream, so that we can test overflowing the queue
        self.flavour = Flavour(4, 64, 48, 0)
        self.stream = send.BytesStream(
            spead2.ThreadPool(),
            send.StreamConfig(rate=1e6, max_heaps=2))
        # A large heap
        ig = send.ItemGroup(flavour=self.flavour)
        ig.add_item(0x1000, 'test', 'A large item', shape=(256 * 1024,), dtype=np.uint8)
        ig['test'].value = np.zeros((256 * 1024,), np.uint8)
        self.heap = ig.get_heap()
        self.threads = []

    def teardown(self):
        for thread in self.threads:
            thread.join()

    def test_overflow(self):
        # Use threads to fill up the first two slots. This is necessary because
        # send_heap is synchronous.
        for i in range(2):
            thread = threading.Thread(target=lambda: self.stream.send_heap(self.heap))
            thread.start()
            self.threads.append(thread)
        # There shouldn't be room now. The sleep is an ugly hack to wait for
        # the threads to enqueue their heaps.
        time.sleep(0.05)
        with pytest.raises(IOError):
            self.stream.send_heap(self.heap)

    def test_send_error(self):
        """An error in sending must be reported."""
        # Create a stream with a packet size that is bigger than the likely
        # MTU. It should cause an error.
        stream = send.UdpStream(
            spead2.ThreadPool(), [("localhost", 8888)],
            send.StreamConfig(max_packet_size=100000), buffer_size=0)
        with pytest.raises(IOError):
            stream.send_heap(self.heap)

    def test_send_explicit_cnt(self):
        """An explicit set heap ID must be respected, and not increment the
        implicit sequence.

        The implicit sequencing is also tested, including wrapping
        """
        ig = send.ItemGroup(flavour=self.flavour)
        self.stream.send_heap(ig.get_start())
        self.stream.set_cnt_sequence(0x1111111111, 0x1234512345)
        self.stream.send_heap(ig.get_start())
        self.stream.send_heap(ig.get_start(), 0x9876543210ab)
        self.stream.send_heap(ig.get_start())
        self.stream.set_cnt_sequence(2**48 - 1, 1)
        self.stream.send_heap(ig.get_start())
        self.stream.send_heap(ig.get_start())
        expected_cnts = [1, 0x1111111111, 0x9876543210ab, 0x2345623456,
                         2**48 - 1, 0]
        expected = b''
        for cnt in expected_cnts:
            expected = b''.join([
                expected,
                self.flavour.make_header(6),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, cnt),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 1),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 1),
                self.flavour.make_immediate(spead2.STREAM_CTRL_ID, spead2.CTRL_STREAM_START),
                self.flavour.make_address(spead2.NULL_ID, 0),
                struct.pack('B', 0)
            ])
        assert hexlify(self.stream.getvalue()) == hexlify(expected)

    def test_invalid_cnt(self):
        """An explicit heap ID that overflows must raise an error."""
        ig = send.ItemGroup(flavour=self.flavour)
        with pytest.raises(IOError):
            self.stream.send_heap(ig.get_start(), 2**48)

    def test_send_heaps_empty(self):
        with pytest.raises(OSError):
            self.stream.send_heaps([], send.GroupMode.ROUND_ROBIN)

    def test_send_heaps_unwind(self):
        """Second or later heap is invalid: must reset original state."""
        ig = send.ItemGroup(flavour=self.flavour)
        self.stream.send_heap(ig.get_start(), substream_index=0)
        with pytest.raises(OSError):
            self.stream.send_heaps(
                [
                    send.HeapReference(ig.get_start(), substream_index=0),
                    send.HeapReference(ig.get_start(), substream_index=100)
                ],
                send.GroupMode.ROUND_ROBIN)
        self.stream.send_heap(ig.get_end(), substream_index=0)

        expected_cnts = [1, 2]
        expected_ctrl = [spead2.CTRL_STREAM_START, spead2.CTRL_STREAM_STOP]
        expected = b''
        for cnt, ctrl in zip(expected_cnts, expected_ctrl):
            expected = b''.join([
                expected,
                self.flavour.make_header(6),
                self.flavour.make_immediate(spead2.HEAP_CNT_ID, cnt),
                self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 1),
                self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
                self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 1),
                self.flavour.make_immediate(spead2.STREAM_CTRL_ID, ctrl),
                self.flavour.make_address(spead2.NULL_ID, 0),
                struct.pack('B', 0)
            ])
        assert hexlify(self.stream.getvalue()) == hexlify(expected)

    def test_interleave(self):
        n = 3
        igs = [send.ItemGroup(flavour=self.flavour) for i in range(n)]
        for i in range(n):
            data = np.arange(i, i + i * 64 + 8).astype(np.uint8)
            igs[i].add_item(0x1000 + i, 'test', 'Test item',
                            shape=data.shape, dtype=data.dtype, value=data)
        self.stream = send.BytesStream(
            spead2.ThreadPool(),
            send.StreamConfig(max_heaps=n, max_packet_size=88))
        self.stream.send_heaps(
            [send.HeapReference(ig.get_heap(descriptors='none', data='all')) for ig in igs],
            send.GroupMode.ROUND_ROBIN)
        expected = b''.join([
            # Heap 0, packet 0 (only packet)
            self.flavour.make_header(5),
            self.flavour.make_immediate(spead2.HEAP_CNT_ID, 1),
            self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 8),
            self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
            self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 8),
            self.flavour.make_address(0x1000, 0),
            igs[0]['test'].value.tobytes(),
            # Heap 1, packet 0
            self.flavour.make_header(5),
            self.flavour.make_immediate(spead2.HEAP_CNT_ID, 2),
            self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 72),
            self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
            self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 40),
            self.flavour.make_address(0x1001, 0),
            igs[1]['test'].value.tobytes()[0 : 40],
            # Heap 2, packet 0
            self.flavour.make_header(5),
            self.flavour.make_immediate(spead2.HEAP_CNT_ID, 3),
            self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 136),
            self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 0),
            self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 40),
            self.flavour.make_address(0x1002, 0),
            igs[2]['test'].value.tobytes()[0 : 40],
            # Heap 1, packet 1
            self.flavour.make_header(4),
            self.flavour.make_immediate(spead2.HEAP_CNT_ID, 2),
            self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 72),
            self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 40),
            self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 32),
            igs[1]['test'].value.tobytes()[40 : 72],
            # Heap 2, packet 1
            self.flavour.make_header(4),
            self.flavour.make_immediate(spead2.HEAP_CNT_ID, 3),
            self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 136),
            self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 40),
            self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 48),
            igs[2]['test'].value.tobytes()[40 : 88],
            # Heap 2, packet 2
            self.flavour.make_header(4),
            self.flavour.make_immediate(spead2.HEAP_CNT_ID, 3),
            self.flavour.make_immediate(spead2.HEAP_LENGTH_ID, 136),
            self.flavour.make_immediate(spead2.PAYLOAD_OFFSET_ID, 88),
            self.flavour.make_immediate(spead2.PAYLOAD_LENGTH_ID, 48),
            igs[2]['test'].value.tobytes()[88 : 136]
        ])
        assert hexlify(self.stream.getvalue()) == hexlify(expected)


class TestTcpStream:
    def test_failed_connect(self):
        with pytest.raises(IOError):
            send.TcpStream(spead2.ThreadPool(), [('127.0.0.1', 8887)])


class TestInprocStream:
    def setup(self):
        self.flavour = Flavour(4, 64, 48, 0)
        self.queue = spead2.InprocQueue()
        self.stream = send.InprocStream(spead2.ThreadPool(), [self.queue])

    def test_stopped_queue(self):
        self.queue.stop()
        ig = send.ItemGroup()
        with pytest.raises(IOError):
            self.stream.send_heap(ig.get_end())


@pytest.mark.skipif(not hasattr(spead2, 'IbvContext'), reason='IBV support not compiled in')
class TestUdpIbvConfig:
    def test_default_construct(self):
        config = send.UdpIbvConfig()
        assert config.endpoints == []
        assert config.interface_address == ''
        assert config.buffer_size == send.UdpIbvConfig.DEFAULT_BUFFER_SIZE
        assert config.ttl == 1
        assert config.comp_vector == 0
        assert config.max_poll == send.UdpIbvConfig.DEFAULT_MAX_POLL
        assert config.memory_regions == []

    def test_kwargs_construct(self):
        data1 = bytearray(10)
        data2 = bytearray(10)
        config = send.UdpIbvConfig(
            endpoints=[('hello', 1234), ('goodbye', 2345)],
            interface_address='1.2.3.4',
            buffer_size=100,
            ttl=2,
            comp_vector=-1,
            max_poll=1000,
            memory_regions=[data1, data2])
        assert config.endpoints == [('hello', 1234), ('goodbye', 2345)]
        assert config.interface_address == '1.2.3.4'
        assert config.buffer_size == 100
        assert config.ttl == 2
        assert config.comp_vector == -1
        assert config.max_poll == 1000
        assert config.memory_regions == [data1, data2]

    def test_default_buffer_size(self):
        config = send.UdpIbvConfig()
        config.buffer_size = 0
        assert config.buffer_size == send.UdpIbvConfig.DEFAULT_BUFFER_SIZE

    def test_bad_max_poll(self):
        config = send.UdpIbvConfig()
        with pytest.raises(ValueError):
            config.max_poll = 0
        with pytest.raises(ValueError):
            config.max_poll = -1

    def test_empty_memory_region(self):
        data = bytearray(0)
        config = send.StreamConfig()
        udp_ibv_config = send.UdpIbvConfig(
            endpoints=[('239.255.88.88', 8888)],
            interface_address='10.0.0.1',
            memory_regions=[data])
        with pytest.raises(ValueError, match='memory region must have non-zero size'):
            send.UdpIbvStream(spead2.ThreadPool(), config, udp_ibv_config)

    def test_overlapping_memory_regions(self):
        data = memoryview(bytearray(10))
        part1 = data[:6]
        part2 = data[5:]
        config = send.StreamConfig()
        udp_ibv_config = send.UdpIbvConfig(
            endpoints=[('239.255.88.88', 8888)],
            interface_address='10.0.0.1',
            memory_regions=[part1, part2])
        with pytest.raises(ValueError, match='memory regions overlap'):
            send.UdpIbvStream(spead2.ThreadPool(), config, udp_ibv_config)

    def test_no_endpoints(self):
        config = send.StreamConfig()
        udp_ibv_config = send.UdpIbvConfig(interface_address='10.0.0.1')
        with pytest.raises(ValueError, match='endpoints is empty'):
            send.UdpIbvStream(spead2.ThreadPool(), config, udp_ibv_config)

    def test_ipv6_endpoints(self):
        config = send.StreamConfig()
        udp_ibv_config = send.UdpIbvConfig(
            endpoints=[('::1', 8888)],
            interface_address='10.0.0.1')
        with pytest.raises(ValueError, match='endpoint is not an IPv4 multicast address'):
            send.UdpIbvStream(spead2.ThreadPool(), config, udp_ibv_config)

    def test_unicast_endpoints(self):
        config = send.StreamConfig()
        udp_ibv_config = send.UdpIbvConfig(
            endpoints=[('10.0.0.1', 8888)],
            interface_address='10.0.0.1')
        with pytest.raises(ValueError, match='endpoint is not an IPv4 multicast address'):
            send.UdpIbvStream(spead2.ThreadPool(), config, udp_ibv_config)

    def test_no_interface_address(self):
        config = send.StreamConfig()
        udp_ibv_config = send.UdpIbvConfig(
            endpoints=[('239.255.88.88', 8888)])
        with pytest.raises(ValueError, match='interface address'):
            send.UdpIbvStream(spead2.ThreadPool(), config, udp_ibv_config)

    def test_bad_interface_address(self):
        config = send.StreamConfig()
        udp_ibv_config = send.UdpIbvConfig(
            endpoints=[('239.255.88.88', 8888)],
            interface_address='this is not an interface address')
        with pytest.raises(RuntimeError, match='Host not found'):
            send.UdpIbvStream(spead2.ThreadPool(), config, udp_ibv_config)

    def test_ipv6_interface_address(self):
        config = send.StreamConfig()
        udp_ibv_config = send.UdpIbvConfig(
            endpoints=[('239.255.88.88', 8888)],
            interface_address='::1')
        with pytest.raises(ValueError, match='interface address'):
            send.UdpIbvStream(spead2.ThreadPool(), config, udp_ibv_config)

    def test_deprecated_constants(self):
        with pytest.deprecated_call():
            assert send.UdpIbvStream.DEFAULT_BUFFER_SIZE == send.UdpIbvConfig.DEFAULT_BUFFER_SIZE
        with pytest.deprecated_call():
            assert send.UdpIbvStream.DEFAULT_MAX_POLL == send.UdpIbvConfig.DEFAULT_MAX_POLL
