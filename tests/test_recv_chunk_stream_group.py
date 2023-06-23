# Copyright 2023 National Research Foundation (SARAO)
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

import threading
import time

import numpy as np
import pytest

import spead2
import spead2.recv as recv
import spead2.send as send

from tests.test_recv_chunk_stream import (
    CHUNK_PAYLOAD_SIZE, HEAP_PAYLOAD_SIZE, HEAPS_PER_CHUNK, place_plain_llc
)

STREAMS = 4


class TestChunkStreamGroupConfig:
    def test_default_construct(self):
        config = recv.ChunkStreamGroupConfig()
        assert config.max_chunks == config.DEFAULT_MAX_CHUNKS
        assert config.eviction_mode == recv.ChunkStreamGroupConfig.EvictionMode.LOSSY

    def test_zero_max_chunks(self):
        with pytest.raises(ValueError):
            recv.ChunkStreamGroupConfig(max_chunks=0)

    def test_max_chunks(self):
        config = recv.ChunkStreamGroupConfig(max_chunks=3)
        assert config.max_chunks == 3
        config.max_chunks = 4
        assert config.max_chunks == 4

    def test_eviction_mode(self):
        EvictionMode = recv.ChunkStreamGroupConfig.EvictionMode
        config = recv.ChunkStreamGroupConfig(eviction_mode=EvictionMode.LOSSLESS)
        assert config.eviction_mode == EvictionMode.LOSSLESS
        config.eviction_mode = EvictionMode.LOSSY
        assert config.eviction_mode == EvictionMode.LOSSY


class TestChunkStreamRingGroup:
    @pytest.fixture
    def data_ring(self):
        return spead2.recv.ChunkRingbuffer(4)

    @pytest.fixture
    def free_ring(self):
        ring = spead2.recv.ChunkRingbuffer(4)
        while not ring.full():
            ring.put(
                recv.Chunk(
                    present=np.zeros(HEAPS_PER_CHUNK, np.uint8),
                    data=np.zeros(CHUNK_PAYLOAD_SIZE, np.uint8)
                )
            )
        return ring

    @pytest.fixture
    def queues(self):
        return [spead2.InprocQueue() for _ in range(STREAMS)]

    @pytest.fixture
    def group(self, data_ring, free_ring):
        group_config = recv.ChunkStreamGroupConfig(max_chunks=4)
        group = recv.ChunkStreamRingGroup(group_config, data_ring, free_ring)
        yield group
        group.stop()

    @pytest.fixture
    def recv_streams(self, queues, group):
        # max_heaps is artificially high to make test_packet_too_old work
        config = spead2.recv.StreamConfig(max_heaps=128)
        chunk_stream_config = spead2.recv.ChunkStreamConfig(
            items=[0x1000, spead2.HEAP_LENGTH_ID],
            max_chunks=4,
            place=place_plain_llc,
        )
        streams = [spead2.recv.ChunkStreamGroupMember(
            spead2.ThreadPool(),
            config=config,
            chunk_stream_config=chunk_stream_config,
            group=group
        ) for _ in queues]
        for stream, queue in zip(streams, queues):
            stream.add_inproc_reader(queue)
        yield streams
        for stream in streams:
            stream.stop()

    @pytest.fixture
    def send_stream(self, queues):
        return send.InprocStream(spead2.ThreadPool(), queues, send.StreamConfig())

    def _send_data(self, send_stream, data, heaps=None):
        """Send the data.

        To send only a subset of heaps (or to send out of order), pass the
        indices to skip in `heaps`.
        """
        data_by_heap = data.reshape(-1, HEAP_PAYLOAD_SIZE)
        ig = spead2.send.ItemGroup()
        ig.add_item(0x1000, 'position', 'position in stream', (), format=[('u', 32)])
        ig.add_item(0x1001, 'payload', 'payload data', (HEAP_PAYLOAD_SIZE,), dtype=np.uint8)
        # Stream groups are impractical to test deterministically, because
        # they rely on concurrent forward progress. So we just feed the
        # data in slowly enough that we expect heaps provided before a
        # sleep to be processed before those after the sleep.
        if heaps is None:
            heaps = range(len(data_by_heap))
        for i in heaps:
            ig['position'].value = i
            ig['payload'].value = data_by_heap[i]
            heap = ig.get_heap(data='all', descriptors='none')
            send_stream.send_heap(heap, substream_index=i % STREAMS)
            time.sleep(0.001)
        # Stop all the queues, which should flush everything and stop the
        # data ring.
        for queue in send_stream.queues:
            queue.stop()

    def test_full_in_order(self, group, queues, recv_streams, send_stream, data_ring, free_ring):
        """Send all the data, in order."""
        chunks = 20
        rng = np.random.default_rng(seed=1)
        data = rng.integers(0, 256, chunks * CHUNK_PAYLOAD_SIZE, np.uint8)
        data_by_chunk = data.reshape(chunks, -1)
        send_thread = threading.Thread(target=self._send_data, args=(send_stream, data))
        send_thread.start()

        for i in range(chunks):
            chunk = data_ring.get()
            assert chunk.chunk_id == i
            np.testing.assert_equal(chunk.present, 1)
            np.testing.assert_equal(chunk.data, data_by_chunk[i])
            group.add_free_chunk(chunk)

        # Stopping all the queues should shut down the data ringbuffer
        with pytest.raises(spead2.Stopped):
            data_ring.get()

        send_thread.join()

    def test_missing_stream(self, group, queues, recv_streams, send_stream, data_ring, free_ring):
        """Skip sending data to one of the streams."""
        chunks = 20
        rng = np.random.default_rng(seed=1)
        data = rng.integers(0, 256, chunks * CHUNK_PAYLOAD_SIZE, np.uint8)
        data_by_heap = data.reshape(chunks, HEAPS_PER_CHUNK, -1)
        heaps = [i for i in range(chunks * HEAPS_PER_CHUNK) if i % STREAMS != 2]
        send_thread = threading.Thread(target=self._send_data, args=(send_stream, data, heaps))
        send_thread.start()

        expected_present = np.ones(chunks * HEAPS_PER_CHUNK, bool)
        expected_present[2::STREAMS] = False
        expected_present = expected_present.reshape(chunks, HEAPS_PER_CHUNK)

        for i in range(chunks):
            chunk = data_ring.get()
            assert chunk.chunk_id == i
            np.testing.assert_equal(chunk.present, expected_present[i])
            actual_data = chunk.data.reshape(HEAPS_PER_CHUNK, -1)
            for j in range(HEAPS_PER_CHUNK):
                if expected_present[i, j]:
                    np.testing.assert_equal(actual_data[j], data_by_heap[i, j])
            group.add_free_chunk(chunk)

        # Stopping all the queues should shut down the data ringbuffer
        with pytest.raises(spead2.Stopped):
            data_ring.get()

        send_thread.join()
