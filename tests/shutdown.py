# Copyright 2017-2018, 2023 National Research Foundation (SARAO)
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

"""Tests for shutdown ordering.

These cannot be run through pytest because they deal with interpreter
shutdown.
"""

import logging

import numpy as np

import spead2
import spead2._spead2
import spead2.recv
import spead2.send


def test_logging_shutdown():
    """Spam the log with lots of messages, then terminate.

    The logging thread needs to be gracefully cleaned up.
    """
    # Set a log level that won't actually display the messages.
    logging.basicConfig(level=logging.ERROR)
    for i in range(20000):
        spead2._spead2.log_info(f"Test message {i}")


def test_running_thread_pool():
    global tp
    tp = spead2.ThreadPool()


def test_running_stream():
    global stream
    logging.basicConfig(level=logging.ERROR)
    stream = spead2.recv.Stream(spead2.ThreadPool())
    stream.add_udp_reader(7148)
    sender = spead2.send.UdpStream(spead2.ThreadPool(), [("localhost", 7148)])
    ig = spead2.send.ItemGroup()
    ig.add_item(
        id=None, name="test", description="test", shape=(), format=[("u", 32)], value=0xDEADBEEF
    )
    heap = ig.get_heap()
    for i in range(5):
        sender.send_heap(heap)


def test_running_chunk_stream_group():
    try:
        import numba
        import scipy
    except ImportError:
        return  # Skip the test if numba/scipy is not available
    from numba import types

    from spead2.numba import intp_to_voidptr
    from spead2.recv.numba import chunk_place_data

    @numba.cfunc(types.void(types.CPointer(chunk_place_data), types.uintp), nopython=True)
    def place(data_ptr, data_size):
        data = numba.carray(data_ptr, 1)
        items = numba.carray(intp_to_voidptr(data[0].items), 2, dtype=np.int64)
        heap_cnt = items[0]
        payload_size = items[1]
        if payload_size == 1024:
            data[0].chunk_id = heap_cnt
            data[0].heap_index = 0
            data[0].heap_offset = 0

    global group
    group = spead2.recv.ChunkStreamRingGroup(
        spead2.recv.ChunkStreamGroupConfig(
            max_chunks=2, eviction_mode=spead2.recv.ChunkStreamGroupConfig.EvictionMode.LOSSLESS
        ),
        spead2.recv.ChunkRingbuffer(4),
        spead2.recv.ChunkRingbuffer(4),
    )
    for _ in range(group.free_ringbuffer.maxsize):
        chunk = spead2.recv.Chunk(data=np.zeros(1024, np.uint8), present=np.zeros(1, np.uint8))
        group.add_free_chunk(chunk)
    place_llc = scipy.LowLevelCallable(place.ctypes, signature="void (void *, size_t)")
    for _ in range(2):
        group.emplace_back(
            spead2.ThreadPool(),
            spead2.recv.StreamConfig(),
            spead2.recv.ChunkStreamConfig(
                items=[spead2.HEAP_CNT_ID, spead2.HEAP_LENGTH_ID], max_chunks=2, place=place_llc
            ),
        )
    queues = [spead2.InprocQueue() for _ in group]
    for queue, stream in zip(queues, group):
        stream.add_inproc_reader(queue)

    # Send 4 chunks of data to the first stream. This will cause it to block
    # while it waits for the second stream to progress.
    send_stream = spead2.send.InprocStream(spead2.ThreadPool(), queues)
    ig = spead2.send.ItemGroup()
    ig.add_item(0x1000, "payload", "payload", shape=(1024,), dtype=np.uint8)
    ig["payload"].value = np.ones(1024, np.uint8)
    heap = ig.get_heap(descriptors="none", data="all")
    for _ in range(4):
        send_stream.send_heap(heap, substream_index=0)
