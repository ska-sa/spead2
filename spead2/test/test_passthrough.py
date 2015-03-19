from __future__ import division, print_function
import numpy as np
import spead2
import spead2.send
import spead2.recv
from nose.tools import *

def assert_items_equal(item1, item2):
    assert_equal(item1.id, item2.id)
    assert_equal(item1.name, item2.name)
    assert_equal(item1.description, item2.description)
    assert_equal(item1.shape, item2.shape)
    assert_equal(item1.format, item2.format)
    assert_equal(item1.dtype, item2.dtype)
    assert_equal(item1.order, item2.order)
    np.testing.assert_equal(item1.value, item2.value)

def assert_item_groups_equal(item_group1, item_group2):
    assert_equal(sorted(item_group1.keys()), sorted(item_group2.keys()))
    for key in item_group1.keys():
        assert_items_equal(item_group1[key], item_group2[key])

class BaseTestPassthrough(object):
    def _test_ig(self, item_group):
        received_item_group = self.transmit_item_group(item_group)
        assert_item_groups_equal(item_group, received_item_group)

    def test_numpy_simple(self):
        ig = spead2.send.ItemGroup()
        data = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.uint16)
        ig.add_item(id=0x2345, name='name', description='description', shape=data.shape, dtype=np.uint16,
                    value=data)
        self._test_ig(ig)

    def transmit_item_group(self, item_group):
        raise NotImplementedError()

class TestPassthroughUdp(BaseTestPassthrough):
    def transmit_item_group(self, item_group):
        thread_pool = spead2.ThreadPool(2)
        sender = spead2.send.UdpStream(
                thread_pool, "localhost", 8888,
                spead2.send.StreamConfig(rate=1e8),
                buffer_size=0)
        receiver = spead2.recv.Stream(thread_pool)
        receiver.add_udp_reader(8888, bind_hostname="localhost")
        gen = spead2.send.HeapGenerator(item_group)
        sender.send_heap(gen.get_heap())
        sender.send_end()
        received_item_group = spead2.ItemGroup()
        for heap in receiver:
            received_item_group.update(heap)
        return received_item_group

class TestPassthroughMem(BaseTestPassthrough):
    def transmit_item_group(self, item_group):
        thread_pool = spead2.ThreadPool(2)
        sender = spead2.send.BytesStream(thread_pool)
        gen = spead2.send.HeapGenerator(item_group)
        sender.send_heap(gen.get_heap())
        sender.send_end()
        receiver = spead2.recv.Stream(thread_pool)
        receiver.add_buffer_reader(sender.getvalue())
        received_item_group = spead2.ItemGroup()
        for heap in receiver:
            received_item_group.update(heap)
        return received_item_group
