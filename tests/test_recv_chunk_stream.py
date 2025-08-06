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

import ctypes
import gc
import struct
import threading
import time
import weakref

import numpy as np
import pytest

import spead2
import spead2.recv as recv
import spead2.send as send

numba = pytest.importorskip("numba")
scipy = pytest.importorskip("scipy")
from numba import types  # noqa: E402

from spead2.numba import intp_to_voidptr  # noqa: E402
from spead2.recv.numba import chunk_place_data  # noqa: E402

HEAP_PAYLOAD_SIZE = 1024
HEAPS_PER_CHUNK = 10
CHUNK_PAYLOAD_SIZE = HEAPS_PER_CHUNK * HEAP_PAYLOAD_SIZE
# These are only applicable to packet presence mode tests
PACKETS_PER_HEAP = 2
PACKET_SIZE = HEAP_PAYLOAD_SIZE // PACKETS_PER_HEAP


def check_refcount(objlist):
    """Check that the objects in the list do not have any other references.

    This is done by making sure that they are garbage collected after removal
    from the list. The caller must make sure to delete any other references
    (e.g. from local variables) before calling. The list is destroyed in the
    process.
    """
    while objlist:
        weak = weakref.ref(objlist.pop())
        # pypy needs multiple garbage collection passes in some cases
        for i in range(10):
            gc.collect()
        assert weak() is None


user_data_type = types.Record.make_c_struct(
    [
        ("scale", types.int_),  # A scale applied to heap_index
        ("placed_heaps_index", types.uintp),  # Index at which to update stats
    ]
)


@numba.cfunc(types.void(types.CPointer(chunk_place_data), types.uintp), nopython=True)
def place_plain(data_ptr, data_size):
    data = numba.carray(data_ptr, 1)
    items = numba.carray(intp_to_voidptr(data[0].items), 2, dtype=np.int64)
    heap_cnt = items[0]
    payload_size = items[1]
    if payload_size == HEAP_PAYLOAD_SIZE:
        data[0].chunk_id = heap_cnt // HEAPS_PER_CHUNK
        data[0].heap_index = heap_cnt % HEAPS_PER_CHUNK
        data[0].heap_offset = data[0].heap_index * HEAP_PAYLOAD_SIZE


@numba.cfunc(
    types.void(types.CPointer(chunk_place_data), types.uintp, types.CPointer(user_data_type)),
    nopython=True,
)
def place_bind(data_ptr, data_size, user_data_ptr):
    # Takes a np.int_ in via user_data to scale the heap index.
    data = numba.carray(data_ptr, 1)
    items = numba.carray(intp_to_voidptr(data[0].items), 3, dtype=np.int64)
    heap_cnt = items[0]
    payload_size = items[1]
    packet_size = items[2]
    user_data = numba.carray(user_data_ptr, 1)
    if payload_size == HEAP_PAYLOAD_SIZE and packet_size == PACKET_SIZE:
        data[0].chunk_id = heap_cnt // HEAPS_PER_CHUNK
        heap_index = heap_cnt % HEAPS_PER_CHUNK
        data[0].heap_index = heap_index * user_data[0].scale
        data[0].heap_offset = heap_index * HEAP_PAYLOAD_SIZE
        batch_stats = numba.carray(
            intp_to_voidptr(data[0].batch_stats),
            user_data[0].placed_heaps_index + 1,
            dtype=np.uint64,
        )
        batch_stats[user_data[0].placed_heaps_index] += 1


@numba.cfunc(
    types.void(types.CPointer(chunk_place_data), types.uintp, types.CPointer(user_data_type)),
    nopython=True,
)
def place_extra(data_ptr, data_size, user_data_ptr):
    # Writes the 'extra' SPEAD item (items[3]) to the extra output array
    data = numba.carray(data_ptr, 1)
    items = numba.carray(intp_to_voidptr(data[0].items), 2, dtype=np.int64)
    extra = numba.carray(intp_to_voidptr(data[0].extra), 1, dtype=np.int64)
    heap_cnt = items[0]
    payload_size = items[1]
    if payload_size == HEAP_PAYLOAD_SIZE:
        data[0].chunk_id = heap_cnt // HEAPS_PER_CHUNK
        data[0].heap_index = heap_cnt % HEAPS_PER_CHUNK
        data[0].heap_offset = data[0].heap_index * HEAP_PAYLOAD_SIZE
        data[0].extra_size = 8  # np.dtype(np.int64).itemsize, but numba doesn't support it
        data[0].extra_offset = data[0].heap_index * data[0].extra_size
        extra[0] = items[2]


# ctypes doesn't distinguish equivalent integer types, so we have to
# specify the signature explicitly.
place_plain_llc = scipy.LowLevelCallable(place_plain.ctypes, signature="void (void *, size_t)")
place_bind_llc = scipy.LowLevelCallable(
    place_bind.ctypes, signature="void (void *, size_t, void *)"
)
place_extra_llc = scipy.LowLevelCallable(place_extra.ctypes, signature="void (void *, size_t)")


class TestChunkStreamConfig:
    def test_default_construct(self):
        config = recv.ChunkStreamConfig()
        assert config.items == []
        assert config.max_chunks == config.DEFAULT_MAX_CHUNKS
        assert config.place is None
        assert config.packet_presence_payload_size == 0

    def test_zero_max_chunks(self):
        config = recv.ChunkStreamConfig()
        with pytest.raises(ValueError):
            config.max_chunks = 0

    def test_set_place_none(self):
        config = recv.ChunkStreamConfig(place=None)
        assert config.place is None

    def test_set_place_empty_tuple(self):
        with pytest.raises(IndexError):
            recv.ChunkStreamConfig(place=())

    def test_set_place_non_tuple(self):
        with pytest.raises(TypeError):
            recv.ChunkStreamConfig(place=1)

    def test_set_place_non_capsule(self):
        with pytest.raises(TypeError):
            recv.ChunkStreamConfig(place=(1,))

    def test_set_place_bad_signature(self):
        place = scipy.LowLevelCallable(place_plain.ctypes, signature="void (void)")
        # One might expect TypeError, but ValueError is what scipy uses for
        # invalid signatures.
        with pytest.raises(ValueError):
            recv.ChunkStreamConfig(place=place)

    def test_set_place_plain(self):
        config = recv.ChunkStreamConfig(place=place_plain_llc)
        assert config.place == place_plain_llc

    def test_set_place_bind(self):
        config = recv.ChunkStreamConfig(place=place_bind_llc)
        assert config.place == place_bind_llc


def make_chunk(label="a"):
    return MyChunk(label, data=bytearray(10), present=bytearray(1))


class TestChunk:
    def test_default_construct(self):
        chunk = recv.Chunk()
        assert chunk.chunk_id == -1
        assert chunk.present is None
        assert chunk.data is None

    def test_set_properties(self):
        buf1 = np.zeros(10, np.uint8)
        buf2 = np.zeros(20, np.uint8)
        chunk = recv.Chunk()
        chunk.chunk_id = 123
        chunk.present = buf1
        chunk.data = buf2
        assert chunk.chunk_id == 123
        assert chunk.present is buf1
        assert chunk.data is buf2
        chunk.present = None
        chunk.data = None
        # Check that we didn't leak any references
        objlist = [buf1, buf2]
        del buf1, buf2
        check_refcount(objlist)


class MyChunk(recv.Chunk):
    """Subclasses Chunk to carry extra metadata."""

    def __init__(self, label, **kwargs):
        super().__init__(**kwargs)
        self.label = label


class TestChunkRingbuffer:
    @pytest.fixture
    def chunk_ringbuffer(self):
        return recv.ChunkRingbuffer(3)

    def test_missing_buffers(self, chunk_ringbuffer):
        with pytest.raises(ValueError):
            chunk_ringbuffer.put(MyChunk("a", present=bytearray(1)))
        with pytest.raises(ValueError):
            chunk_ringbuffer.put(MyChunk("a", data=bytearray(1)))

    def test_qsize(self, chunk_ringbuffer):
        """Test qsize, maxsize, empty and full."""
        assert chunk_ringbuffer.empty()
        assert not chunk_ringbuffer.full()
        assert chunk_ringbuffer.qsize() == 0
        assert chunk_ringbuffer.maxsize == 3

        chunk_ringbuffer.put(make_chunk())
        assert not chunk_ringbuffer.empty()
        assert not chunk_ringbuffer.full()
        assert chunk_ringbuffer.qsize() == 1

        for label in ["b", "c"]:
            chunk_ringbuffer.put(make_chunk(label))
        assert not chunk_ringbuffer.empty()
        assert chunk_ringbuffer.full()
        assert chunk_ringbuffer.qsize() == 3

        for i in range(3):
            chunk_ringbuffer.get()
        assert chunk_ringbuffer.empty()

    @pytest.mark.parametrize("method", [recv.ChunkRingbuffer.get, recv.ChunkRingbuffer.get_nowait])
    def test_stop(self, chunk_ringbuffer, method):
        chunk_ringbuffer.put(make_chunk())
        chunk_ringbuffer.stop()
        chunk_ringbuffer.get()  # Should get the item in the queue
        with pytest.raises(spead2.Stopped):
            method(chunk_ringbuffer)

    def test_round_trip(self, chunk_ringbuffer):
        chunk = make_chunk()
        data = chunk.data
        present = chunk.present
        chunk_ringbuffer.put(chunk)
        out_chunk = chunk_ringbuffer.get()
        assert out_chunk is chunk
        assert out_chunk.label == "a"
        assert out_chunk.data is data
        assert out_chunk.present is present
        # Check that we haven't leaked a reference
        objlist = [chunk]
        del chunk, out_chunk
        check_refcount(objlist)

    def test_get_nowait(self, chunk_ringbuffer):
        with pytest.raises(spead2.Empty):
            chunk_ringbuffer.get_nowait()
        chunk = make_chunk()
        chunk_ringbuffer.put(chunk)
        assert chunk_ringbuffer.get_nowait() is chunk

    def test_put_nowait(self, chunk_ringbuffer):
        for label in ["a", "b", "c"]:
            chunk_ringbuffer.put_nowait(make_chunk(label))
        d = make_chunk("d")
        with pytest.raises(spead2.Full):
            chunk_ringbuffer.put_nowait(d)
        # put_nowait moves the data into a chunk_wrapper. Check that a failed
        # put has no side effects.
        assert d.data is not None
        assert d.present is not None
        for label in ["a", "b", "c"]:
            assert chunk_ringbuffer.get().label == label

    def test_iterate(self, chunk_ringbuffer):
        for label in ["a", "b", "c"]:
            chunk_ringbuffer.put_nowait(make_chunk(label))
        chunk_ringbuffer.stop()
        out_labels = [chunk.label for chunk in chunk_ringbuffer]
        assert out_labels == ["a", "b", "c"]

    def test_block(self, chunk_ringbuffer):
        def thread_code():
            time.sleep(0.05)
            chunk_ringbuffer.put(make_chunk())

        thread = threading.Thread(target=thread_code)
        thread.start()
        with pytest.raises(spead2.Empty):
            chunk_ringbuffer.get_nowait()
        chunk = chunk_ringbuffer.get()
        assert chunk.label == "a"
        thread.join()

    def test_add_remove_producer(self, chunk_ringbuffer):
        chunk_ringbuffer.add_producer()
        chunk_ringbuffer.add_producer()
        # Removing first producer should not stop the ringbuffer
        assert not chunk_ringbuffer.remove_producer()
        with pytest.raises(spead2.Empty):
            chunk_ringbuffer.get_nowait()
        # Removing the second should stop it
        assert chunk_ringbuffer.remove_producer()
        with pytest.raises(spead2.Stopped):
            chunk_ringbuffer.get_nowait()


class TestChunkRingStream:
    @pytest.fixture
    def data_ring(self):
        return spead2.recv.ChunkRingbuffer(4)

    @pytest.fixture
    def free_ring(self):
        ring = spead2.recv.ChunkRingbuffer(6)
        while not ring.full():
            ring.put(
                recv.Chunk(
                    present=np.zeros(HEAPS_PER_CHUNK, np.uint8),
                    data=np.zeros(CHUNK_PAYLOAD_SIZE, np.uint8),
                    extra=np.zeros(HEAPS_PER_CHUNK, np.int64),
                )
            )
        return ring

    @pytest.fixture
    def item_group(self):
        ig = spead2.send.ItemGroup()
        ig.add_item(0x1000, "position", "position in stream", (), format=[("u", 32)])
        ig.add_item(0x1001, "payload", "payload data", (HEAP_PAYLOAD_SIZE,), dtype=np.uint8)
        ig.add_item(0x1002, "extra", "extra data to capture", (), format=[("u", 32)])
        return ig

    @pytest.fixture
    def queue(self):
        return spead2.InprocQueue()

    # Can be overridden to change the config for individual fixtures
    @pytest.fixture
    def config_kwargs(self):
        return {}

    @pytest.fixture
    def recv_stream(self, request, data_ring, free_ring, queue, config_kwargs):
        config_kwargs.setdefault("place", place_plain_llc)
        stream = spead2.recv.ChunkRingStream(
            spead2.ThreadPool(),
            # max_heaps is artificially high to make test_packet_too_old work
            spead2.recv.StreamConfig(max_heaps=128),
            spead2.recv.ChunkStreamConfig(
                items=[0x1000, spead2.HEAP_LENGTH_ID, 0x1002],
                max_chunks=4,
                max_heap_extra=np.dtype(np.int64).itemsize,
                **config_kwargs,
            ),
            data_ring,
            free_ring,
        )
        stream.add_inproc_reader(queue)
        yield stream
        stream.stop()

    @pytest.fixture
    def send_stream(self, queue):
        return send.InprocStream(spead2.ThreadPool(), [queue], send.StreamConfig())

    def make_heap_payload(self, position):
        """Construct payload data for a heap.

        The payload is constructed in 4-byte pieces, where the first two
        bytes are `position` and the second two increment with each 4-byte
        piece in the packet.
        """
        heap_payload = np.zeros(HEAP_PAYLOAD_SIZE // 2, np.uint16)
        heap_payload[0::2] = position
        heap_payload[1::2] = range(HEAP_PAYLOAD_SIZE // 4)
        return heap_payload.view(np.uint8)

    @pytest.mark.parametrize("send_end", [True, False])
    def test_cleanup(self, send_stream, recv_stream, item_group, send_end):
        """Send some heaps and don't retrieve the chunks, making sure cleanup works."""
        send_stream.send_heap(item_group.get_heap(descriptors="all", data="none"))
        for i in range(1000):
            item_group["position"].value = i
            item_group["payload"].value = self.make_heap_payload(i)
            send_stream.send_heap(item_group.get_heap(descriptors="none", data="all"))
        if send_end:
            send_stream.send_heap(item_group.get_end())

    def send_heaps(self, send_stream, item_group, positions):
        """Send heaps with given numbers."""
        send_stream.send_heap(item_group.get_heap(descriptors="all", data="none"))
        for i in positions:
            item_group["position"].value = i
            item_group["payload"].value = self.make_heap_payload(i)
            item_group["extra"].value = i ^ 0xBEEF
            send_stream.send_heap(item_group.get_heap(descriptors="none", data="all"))
        send_stream.send_heap(item_group.get_end())

    def check_chunk(self, chunk, expected_chunk_id, expected_present, extra=False):
        """Validate a chunk."""
        assert chunk.chunk_id == expected_chunk_id
        assert chunk.present.dtype == np.dtype(np.uint8)
        np.testing.assert_equal(chunk.present, expected_present)
        for i, p in enumerate(chunk.present):
            if p:
                position = chunk.chunk_id * HEAPS_PER_CHUNK + i
                np.testing.assert_equal(
                    chunk.data[i * HEAP_PAYLOAD_SIZE : (i + 1) * HEAP_PAYLOAD_SIZE],
                    self.make_heap_payload(position),
                )
                if extra:
                    assert chunk.extra[i] == position ^ 0xBEEF

    def check_chunk_packets(self, chunk, expected_chunk_id, expected_present):
        """Validate a chunk from test_packet_presence."""
        assert chunk.chunk_id == expected_chunk_id
        assert chunk.present.dtype == np.dtype(np.uint8)
        np.testing.assert_equal(chunk.present, expected_present)
        for i, p in enumerate(chunk.present):
            if p:
                heap_index = chunk.chunk_id * HEAPS_PER_CHUNK + i // PACKETS_PER_HEAP
                packet_index = i % PACKETS_PER_HEAP
                start = packet_index * PACKET_SIZE
                end = (packet_index + 1) * PACKET_SIZE
                np.testing.assert_equal(
                    chunk.data[i * PACKET_SIZE : (i + 1) * PACKET_SIZE],
                    self.make_heap_payload(heap_index)[start:end],
                )

    @pytest.mark.parametrize(
        "config_kwargs, extra",
        [(dict(place=place_plain_llc), False), (dict(place=place_extra_llc), True)],
    )
    def test_basic(self, send_stream, recv_stream, item_group, extra):
        n_heaps = 103
        self.send_heaps(send_stream, item_group, range(n_heaps))
        seen = 0
        for i, chunk in enumerate(recv_stream.data_ringbuffer):
            expected_present = np.ones(HEAPS_PER_CHUNK, np.uint8)
            if i == n_heaps // HEAPS_PER_CHUNK:
                # It's the last chunk
                expected_present[n_heaps % HEAPS_PER_CHUNK :] = 0
            self.check_chunk(chunk, i, expected_present, extra)
            seen += 1
            recv_stream.add_free_chunk(chunk)
        assert seen == n_heaps // HEAPS_PER_CHUNK + 1

    def test_out_of_order(self, send_stream, recv_stream, item_group):
        """Test some heaps out of chunk order, but within the window."""
        pos = [37, 7, 27, 47, 17, 87, 57, 77, 67]
        self.send_heaps(send_stream, item_group, pos)

        seen = 0
        expected_present = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], np.uint8)
        for i, chunk in enumerate(recv_stream.data_ringbuffer):
            self.check_chunk(chunk, i, expected_present)
            seen += 1
            recv_stream.add_free_chunk(chunk)
        assert seen == len(pos)

    def test_jump(self, send_stream, recv_stream, item_group):
        """Test discontiguous jump in chunks."""
        pos = [100, 200]
        expected_chunks = [7, 8, 9, 10, 17, 18, 19, 20]
        self.send_heaps(send_stream, item_group, pos)
        seen = 0
        for i, chunk in enumerate(recv_stream.data_ringbuffer):
            expected_present = np.zeros(HEAPS_PER_CHUNK, np.uint8)
            if expected_chunks[i] * HEAPS_PER_CHUNK in pos:
                expected_present[0] = 1
            self.check_chunk(chunk, expected_chunks[i], expected_present)
            seen += 1
            recv_stream.add_free_chunk(chunk)
        assert seen == len(expected_chunks)

    def test_heap_too_old(self, send_stream, recv_stream, item_group):
        """Test a heap arriving too late."""
        pos = list(range(10, 41)) + [0] + list(range(41, 50))  # 0 is just too late
        self.send_heaps(send_stream, item_group, pos)
        seen = 0
        for i, chunk in enumerate(recv_stream.data_ringbuffer):
            expected_present = np.ones(HEAPS_PER_CHUNK, np.uint8)
            if i == 0:
                expected_present[:] = 0
            self.check_chunk(chunk, i, expected_present)
            seen += 1
            recv_stream.add_free_chunk(chunk)
        assert seen == 5  # Will see chunk 0 with no heaps, but won't see it again
        recv_stream.stop()  # Ensure that stats are brought up to date
        assert recv_stream.stats["too_old_heaps"] == 1
        assert recv_stream.stats["rejected_heaps"] == 2  # Descriptors and stop heap

    @pytest.mark.parametrize("config_kwargs", [dict(place=place_extra_llc)])
    def test_extra(self, send_stream, recv_stream, item_group):
        """Test writing extra data about each heap."""
        n_heaps = 103
        self.send_heaps(send_stream, item_group, range(n_heaps))
        seen = 0
        for i, chunk in enumerate(recv_stream.data_ringbuffer):
            expected_present = np.ones(HEAPS_PER_CHUNK, np.uint8)
            if i == n_heaps // HEAPS_PER_CHUNK:
                # It's the last chunk
                expected_present[n_heaps % HEAPS_PER_CHUNK :] = 0
            self.check_chunk(chunk, i, expected_present)
            seen += 1
            recv_stream.add_free_chunk(chunk)
        assert seen == n_heaps // HEAPS_PER_CHUNK + 1

    def test_shared_ringbuffer(self, send_stream, recv_stream, item_group):
        recv_stream2 = spead2.recv.ChunkRingStream(
            spead2.ThreadPool(),
            spead2.recv.StreamConfig(stream_id=1),
            spead2.recv.ChunkStreamConfig(
                items=[0x1000, spead2.HEAP_LENGTH_ID], max_chunks=4, place=place_plain_llc
            ),
            recv_stream.data_ringbuffer,
            recv_stream.free_ringbuffer,
        )
        queue2 = spead2.InprocQueue()
        recv_stream2.add_inproc_reader(queue2)
        send_stream2 = send.InprocStream(spead2.ThreadPool(), [queue2], send.StreamConfig())

        n_heaps = [17, 23]
        self.send_heaps(send_stream, item_group, range(n_heaps[0]))
        self.send_heaps(send_stream2, item_group, range(n_heaps[1]))
        seen = [0, 0]
        for chunk in recv_stream.data_ringbuffer:
            assert 0 <= chunk.stream_id < 2
            expected_present = np.ones(HEAPS_PER_CHUNK, np.uint8)
            if chunk.chunk_id == n_heaps[chunk.stream_id] // HEAPS_PER_CHUNK:
                # It's the last chunk for the stream
                expected_present[n_heaps[chunk.stream_id] % HEAPS_PER_CHUNK :] = 0
            self.check_chunk(chunk, seen[chunk.stream_id], expected_present)
            seen[chunk.stream_id] += 1
            recv_stream.add_free_chunk(chunk)
        assert seen[0] == n_heaps[0] // HEAPS_PER_CHUNK + 1
        assert seen[1] == n_heaps[1] // HEAPS_PER_CHUNK + 1

    def test_missing_place_callback(self, data_ring, free_ring):
        with pytest.raises(ValueError):
            spead2.recv.ChunkRingStream(
                spead2.ThreadPool(),
                spead2.recv.StreamConfig(),
                spead2.recv.ChunkStreamConfig(items=[0x1000, spead2.HEAP_LENGTH_ID]),
                data_ring,
                free_ring,
            )

    def make_packet(self, position, start, end):
        """Construct a single packet.

        Parameters
        ----------
        position
            Value of the "position" immediate item
        start
            First heap payload byte in this packet
        end
            Last heap payload byte in this packet (exclusive)
        """
        assert 0 <= start < end <= HEAP_PAYLOAD_SIZE
        heap_payload = self.make_heap_payload(position)
        parts = [
            # Magic, version, item ID bytes, heap address bytes, flags, number of items
            struct.pack(">BBBBHH", 0x53, 4, 2, 6, 0, 6),
            # Item ID (and immediate flag), item value/offset
            struct.pack(">HxxI", 0x8000 | spead2.HEAP_CNT_ID, position),
            struct.pack(">HxxI", 0x8000 | spead2.PAYLOAD_OFFSET_ID, start),
            struct.pack(">HxxI", 0x8000 | spead2.PAYLOAD_LENGTH_ID, end - start),
            struct.pack(">HxxI", 0x8000 | spead2.HEAP_LENGTH_ID, HEAP_PAYLOAD_SIZE),
            struct.pack(">HxxI", 0x8000 | 0x1000, position),
            struct.pack(">HxxI", 0x1001, 0),
            heap_payload[start:end].tobytes(),
        ]
        return b"".join(parts)

    def test_packet_too_old(self, recv_stream, queue):
        """Test a packet that adds to an existing heap whose chunk was already aged out."""
        # Start a heap
        queue.add_packet(self.make_packet(0, 0, 100))
        # Age out the chunk by making a new one and filling it
        for pos in range(40, 50):
            queue.add_packet(self.make_packet(pos, 0, HEAP_PAYLOAD_SIZE))
        # Finish the heap we started earlier
        queue.add_packet(self.make_packet(0, 100, HEAP_PAYLOAD_SIZE))
        # Add another chunk, so that we can validate that we didn't just stop
        # with heap 0.
        for pos in range(50, 60):
            queue.add_packet(self.make_packet(pos, 0, HEAP_PAYLOAD_SIZE))
        queue.stop()

        seen = 0
        for i, chunk in enumerate(recv_stream.data_ringbuffer):
            expected_present = np.zeros(HEAPS_PER_CHUNK, np.uint8)
            if i >= 4:
                expected_present[:] = 1
            self.check_chunk(chunk, i, expected_present)
            seen += 1
            recv_stream.add_free_chunk(chunk)
        assert seen == 6

    def test_packet_presence(self, data_ring, queue):
        """Test packet presence feature."""
        # Each heap is split into two packets. Create a free ring where the
        # chunks have space for this.
        free_ring = spead2.recv.ChunkRingbuffer(4)
        while not free_ring.full():
            free_ring.put(
                recv.Chunk(
                    present=np.zeros(HEAPS_PER_CHUNK * PACKETS_PER_HEAP, np.uint8),
                    data=np.zeros(CHUNK_PAYLOAD_SIZE, np.uint8),
                )
            )

        stream_config = spead2.recv.StreamConfig(max_heaps=128, allow_out_of_order=True)
        # Note: user_data is deliberately not assigned to a local variable, so
        # that reference-counting errors are more likely to be detected.
        user_data = np.zeros(1, dtype=user_data_type.dtype)
        user_data["scale"] = PACKETS_PER_HEAP
        user_data["placed_heaps_index"] = stream_config.add_stat("placed_heaps")
        place_bind_llc = scipy.LowLevelCallable(
            place_bind.ctypes,
            user_data=user_data.ctypes.data_as(ctypes.c_void_p),
            signature="void (void *, size_t, void *)",
        )
        stream = spead2.recv.ChunkRingStream(
            spead2.ThreadPool(),
            stream_config,
            spead2.recv.ChunkStreamConfig(
                items=[0x1000, spead2.HEAP_LENGTH_ID, spead2.PAYLOAD_LENGTH_ID],
                max_chunks=4,
                place=place_bind_llc,
            ).enable_packet_presence(PACKET_SIZE),
            data_ring,
            free_ring,
        )
        stream.add_inproc_reader(queue)

        queue.add_packet(self.make_packet(4, 0, PACKET_SIZE))
        queue.add_packet(self.make_packet(4, PACKET_SIZE, 2 * PACKET_SIZE))
        queue.add_packet(self.make_packet(17, PACKET_SIZE, 2 * PACKET_SIZE))
        queue.stop()
        chunks = list(stream.data_ringbuffer)
        assert len(chunks) == 2
        self.check_chunk_packets(
            chunks[0],
            0,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.uint8),
        )
        self.check_chunk_packets(
            chunks[1],
            1,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], np.uint8),
        )
        assert stream.stats["placed_heaps"] == 2

    @pytest.mark.parametrize("config_kwargs", [{"pop_if_full": True}])
    def test_pop_if_full(self, send_stream, recv_stream, item_group):
        assert recv_stream.stats["discarded_chunks"] == 0
        n_chunks = 6
        n_heaps = n_chunks * HEAPS_PER_CHUNK
        start = n_chunks - recv_stream.data_ringbuffer.maxsize
        assert start > 0
        self.send_heaps(send_stream, item_group, range(n_heaps))
        # pop_if_full is fundamentally non-deterministic, so we need to
        # give it some time for the data to percolate through.
        time.sleep(0.1)
        for i, chunk in enumerate(recv_stream.data_ringbuffer, start=start):
            # expected_present = np.ones(HEAPS_PER_CHUNK, np.uint8)
            # self.check_chunk(chunk, i, expected_present, False)
            print(chunk.chunk_id)
            recv_stream.add_free_chunk(chunk)
        assert recv_stream.stats["discarded_chunks"] == start
