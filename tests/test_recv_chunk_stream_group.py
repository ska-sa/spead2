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

    def test_zero_max_chunks(self):
        with pytest.raises(ValueError):
            recv.ChunkStreamGroupConfig(max_chunks=0)

    def test_max_chunks(self):
        config = recv.ChunkStreamGroupConfig(max_chunks=3)
        assert config.max_chunks == 3
        config.max_chunks = 4
        assert config.max_chunks == 4


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

    @pytest.fixture
    def item_group(self):
        ig = spead2.send.ItemGroup()
        ig.add_item(0x1000, 'position', 'position in stream', (), format=[('u', 32)])
        ig.add_item(0x1001, 'payload', 'payload data', (HEAP_PAYLOAD_SIZE,), dtype=np.uint8)
        return ig

    def test_full_in_order(self, group, queues, recv_streams, send_stream, data_ring, free_ring, item_group):
        chunks = 20
        rng = np.random.default_rng(seed=1)
        data = rng.integers(0, 256, chunks * CHUNK_PAYLOAD_SIZE, np.uint8)
        data_by_chunk = data.reshape(chunks, -1)
        data_by_heap = data.reshape(chunks * HEAPS_PER_CHUNK, -1)

        # Stream groups are impractical to test deterministically, because
        # they rely on concurrent forward progress. So we just feed the
        # data in slowly enough that we expect heaps provided before a
        # sleep to be processed before those after the sleep.
        def send_data():
            for i, payload in enumerate(data_by_heap):
                if i % STREAMS == 0:
                    time.sleep(0.005)
                item_group['position'].value = i
                item_group['payload'].value = payload
                heap = item_group.get_heap(data='all', descriptors='none')
                send_stream.send_heap(heap, substream_index=i % STREAMS)
            # Stop all the queues, which should flush everything and stop the data
            # ring.
            time.sleep(0.01)
            for queue in queues:
                queue.stop()

        send_thread = threading.Thread(target=send_data)
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
