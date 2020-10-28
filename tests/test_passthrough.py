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

"""Test that data can be passed over the SPEAD protocol using the various transports."""

import os
import socket
import sys

import numpy as np
import netifaces
import pytest

import spead2
import spead2.send
import spead2.recv


def _assert_items_equal(item1, item2):
    assert item1.id == item2.id
    assert item1.name == item2.name
    assert item1.description == item2.description
    assert item1.shape == item2.shape
    assert item1.format == item2.format
    # Byte order need not match, provided that values are received correctly
    if item1.dtype is not None and item2.dtype is not None:
        assert item1.dtype.newbyteorder('<') == item2.dtype.newbyteorder('<')
    else:
        assert item1.dtype == item2.dtype
    assert item1.order == item2.order
    # Comparing arrays has many issues. Convert them to lists where appropriate
    value1 = item1.value
    value2 = item2.value
    if hasattr(value1, 'tolist'):
        value1 = value1.tolist()
    if hasattr(value2, 'tolist'):
        value2 = value2.tolist()
    assert value1 == value2


def assert_item_groups_equal(item_group1, item_group2):
    assert sorted(item_group1.keys()) == sorted(item_group2.keys())
    for key in item_group1.keys():
        _assert_items_equal(item_group1[key], item_group2[key])


@pytest.mark.timeout(5)
class BaseTestPassthrough:
    """Tests common to all transports and libraries"""

    is_lossy = False
    requires_ipv6 = False

    @classmethod
    def check_ipv6(cls):
        if not socket.has_ipv6:
            pytest.skip('platform does not support IPv6')
        # Travis build systems fail to bind to an IPv6 address
        sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        try:
            sock.bind(("::1", 8888))
        except OSError:
            pytest.skip('platform cannot bind IPv6 localhost address')
        finally:
            sock.close()

    def _test_item_groups(self, item_groups, *,
                          memcpy=spead2.MEMCPY_STD, allocator=None,
                          new_order='=', round_robin=False):
        received_item_groups = self.transmit_item_groups(
            item_groups,
            memcpy=memcpy, allocator=allocator,
            new_order=new_order, round_robin=round_robin)
        assert len(received_item_groups) == len(item_groups)
        for received_item_group, item_group in zip(received_item_groups, item_groups):
            assert_item_groups_equal(item_group, received_item_group)
            for item in received_item_group.values():
                if item.dtype is not None:
                    assert item.value.dtype == item.value.dtype.newbyteorder(new_order)

    def _test_item_group(self, item_group, *,
                         memcpy=spead2.MEMCPY_STD, allocator=None,
                         new_order='=', round_robin=False):
        self._test_item_groups(
            [item_group],
            memcpy=memcpy,
            allocator=allocator,
            new_order=new_order,
            round_robin=round_robin)

    def test_numpy_simple(self):
        """A basic array with numpy encoding"""
        ig = spead2.send.ItemGroup()
        data = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.uint16)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=data.shape, dtype=data.dtype, value=data)
        self._test_item_group(ig)

    def test_numpy_byteorder(self):
        """A numpy array in non-native byte order"""
        ig = spead2.send.ItemGroup()
        data = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.dtype(np.uint16).newbyteorder())
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=data.shape, dtype=data.dtype, value=data)
        self._test_item_group(ig)
        self._test_item_group(ig, new_order='|')

    def test_numpy_large(self):
        """A numpy style array split across several packets. It also
        uses non-temporal copies, a custom allocator, and a memory pool,
        to test that those all work."""
        # macOS doesn't have a big enough socket buffer to reliably transmit
        # the whole thing over UDP.
        if self.is_lossy and sys.platform == 'darwin':
            pytest.skip("macOS can't reliably handle large heaps over UDP")
        ig = spead2.send.ItemGroup()
        data = np.random.randn(100, 200)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=data.shape, dtype=data.dtype, value=data)
        allocator = spead2.MmapAllocator()
        pool = spead2.MemoryPool(1, 4096, 4, 4, allocator)
        self._test_item_group(ig, memcpy=spead2.MEMCPY_NONTEMPORAL, allocator=pool)

    def test_fallback_struct_whole_bytes(self):
        """A structure with non-byte-aligned elements, but which is
        byte-aligned overall."""
        ig = spead2.send.ItemGroup()
        format = [('u', 4), ('f', 64), ('i', 4)]
        data = (12, 1.5, -3)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(), format=format, value=data)
        self._test_item_group(ig)

    def test_string(self):
        """Byte string is converted to array of characters and back."""
        ig = spead2.send.ItemGroup()
        format = [('c', 8)]
        data = 'Hello world'
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(None,), format=format, value=data)
        self._test_item_group(ig)

    def test_fallback_array_partial_bytes_small(self):
        """An array which takes a fractional number of bytes per element
        and is small enough to encode in an immediate.
        """
        ig = spead2.send.ItemGroup()
        format = [('u', 7)]
        data = [127, 12, 123]
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(len(data),), format=format, value=data)
        self._test_item_group(ig)

    def test_fallback_types(self):
        """An array structure using a mix of types."""
        ig = spead2.send.ItemGroup()
        format = [('b', 1), ('i', 7), ('c', 8), ('f', 32)]
        data = [(True, 17, b'y', 1.0), (False, -23, b'n', -1.0)]
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(2,), format=format, value=data)
        self._test_item_group(ig)

    def test_numpy_fallback_struct(self):
        """A structure specified using a format, but which is encodable using numpy."""
        ig = spead2.send.ItemGroup()
        format = [('u', 8), ('f', 32)]
        data = (12, 1.5)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(), format=format, value=data)
        self._test_item_group(ig)

    def test_fallback_struct_partial_bytes(self):
        """A structure which takes a fractional number of bytes per element."""
        ig = spead2.send.ItemGroup()
        format = [('u', 4), ('f', 64)]
        data = (12, 1.5)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(), format=format, value=data)
        self._test_item_group(ig)

    def test_fallback_scalar(self):
        """Send a scalar using fallback format descriptor."""
        ig = spead2.send.ItemGroup()
        format = [('f', 64)]
        data = 1.5
        ig.add_item(id=0x2345, name='scalar name', description='scalar description',
                    shape=(), format=format, value=data)
        self._test_item_group(ig)

    def test_many_items(self):
        """Sends many items.

        The implementation handles few-item heaps differently (for
        performance), so need to test that the many-item code works too.
        """
        ig = spead2.send.ItemGroup()
        for i in range(50):
            name = f'test item {i}'
            ig.add_item(id=0x2345 + i, name=name, description=name,
                        shape=(), format=[('u', 40)], value=0x12345 * i)
        self._test_item_group(ig)

    def transmit_item_groups(self, item_groups, *,
                             memcpy, allocator, new_order='=', round_robin=False):
        """Transmit `item_groups` over the chosen transport.

        Return the item groups received at the other end. Each item group will
        be transmitted over a separate substream (thus, the transport must
        support substreams if `item_groups` has more than one element).
        """
        if self.requires_ipv6:
            self.check_ipv6()
        recv_config = spead2.recv.StreamConfig(memcpy=memcpy)
        if allocator is not None:
            recv_config.memory_allocator = allocator
        receivers = [
            spead2.recv.Stream(spead2.ThreadPool(), recv_config)
            for i in range(len(item_groups))
        ]
        self.prepare_receivers(receivers)
        sender = self.prepare_senders(spead2.ThreadPool(), len(item_groups))
        gens = [
            spead2.send.HeapGenerator(item_group)
            for item_group in item_groups
        ]
        if len(item_groups) != 1:
            # Use reversed order so that if everything is actually going
            # through the same transport it will get picked up.
            if round_robin:
                sender.send_heaps(
                    [
                        spead2.send.HeapReference(gen.get_heap(), substream_index=i)
                        for i, gen in reversed(list(enumerate(gens)))
                    ],
                    spead2.send.GroupMode.ROUND_ROBIN
                )
                sender.send_heaps(
                    [
                        spead2.send.HeapReference(gen.get_end(), substream_index=i)
                        for i, gen in enumerate(gens)
                    ],
                    spead2.send.GroupMode.ROUND_ROBIN
                )
            else:
                for i, gen in reversed(list(enumerate(gens))):
                    sender.send_heap(gen.get_heap(), substream_index=i)
                for i, gen in enumerate(gens):
                    sender.send_heap(gen.get_end(), substream_index=i)
        else:
            # This is a separate code path to give coverage of the case where
            # the substream index is implicit.
            sender.send_heap(gens[0].get_heap())
            sender.send_heap(gens[0].get_end())
        received_item_groups = []
        for receiver in receivers:
            ig = spead2.ItemGroup()
            for heap in receiver:
                ig.update(heap, new_order)
            received_item_groups.append(ig)
        return received_item_groups

    def prepare_receiver(self, receiver):
        """Generate a single receiver to use in the test.

        This should be implemented by classes that don't support substreams.
        """
        raise NotImplementedError()

    def prepare_sender(self, thread_pool):
        """Generate a sender with a single substream.

        This should be implemented by classes that don't support substreams.
        """
        raise NotImplementedError()

    def prepare_receivers(self, receivers):
        """Generate receivers to use in the test."""
        assert len(receivers) == 1
        self.prepare_receiver(receivers[0])

    def prepare_senders(self, thread_pool, n):
        """Generate a sender to use in the test, with `n` substreams."""
        assert n == 1
        return self.prepare_sender(thread_pool)


class BaseTestPassthroughSubstreams(BaseTestPassthrough):
    """Tests for send stream classes that support multiple substreams."""

    def test_substreams(self):
        item_groups = []
        for i in range(4):
            ig = spead2.ItemGroup()
            ig.add_item(id=0x2345, name='int', description='an integer',
                        shape=(), format=[('i', 32)], value=i)
            item_groups.append(ig)
        self._test_item_groups(item_groups)

    @pytest.mark.parametrize('size', [10, 20000])
    def test_round_robin(self, size):
        # The interleaving and substream features are independent, but the
        # test framework is set up for one item group per substream.
        item_groups = []
        for i in range(4):
            value = np.random.randint(0, 256, size=size).astype(np.uint8)
            ig = spead2.ItemGroup()
            ig.add_item(id=0x2345, name='arr', description='a random array',
                        shape=(size,), dtype='u8', value=value)
            item_groups.append(ig)
        self._test_item_groups(item_groups, round_robin=True)

    def test_round_robin_mixed_sizes(self):
        sizes = [20000, 2000, 40000, 30000]
        item_groups = []
        for size in sizes:
            value = np.random.randint(0, 256, size=size).astype(np.uint8)
            ig = spead2.ItemGroup()
            ig.add_item(id=0x2345, name='arr', description='a random array',
                        shape=(size,), dtype='u8', value=value)
            item_groups.append(ig)
        self._test_item_groups(item_groups, round_robin=True)

    def prepare_receivers(self, receivers):
        raise NotImplementedError()

    def prepare_senders(self, thread_pool, n):
        raise NotImplementedError()


class TestPassthroughUdp(BaseTestPassthroughSubstreams):
    is_lossy = True

    def prepare_receivers(self, receivers):
        for i, receiver in enumerate(receivers):
            receiver.add_udp_reader(8888 + i, bind_hostname="localhost")

    def prepare_senders(self, thread_pool, n):
        if n == 1:
            with pytest.deprecated_call():
                return spead2.send.UdpStream(
                    thread_pool, "localhost", 8888,
                    spead2.send.StreamConfig(rate=1e7),
                    buffer_size=0)
        else:
            return spead2.send.UdpStream(
                thread_pool,
                [("localhost", 8888 + i) for i in range(n)],
                spead2.send.StreamConfig(rate=1e7),
                buffer_size=0)

    def test_empty_endpoints(self):
        with pytest.raises(ValueError):
            spead2.send.UdpStream(
                spead2.ThreadPool(), [], spead2.send.StreamConfig(rate=1e7))

    def test_mixed_protocols(self):
        with pytest.raises(ValueError):
            spead2.send.UdpStream(
                spead2.ThreadPool(),
                [('127.0.0.1', 8888), ('::1', 8888)],
                spead2.send.StreamConfig(rate=1e7))


class TestPassthroughUdp6(BaseTestPassthroughSubstreams):
    is_lossy = True
    requires_ipv6 = True

    def prepare_receivers(self, receivers):
        for i, receiver in enumerate(receivers):
            receiver.add_udp_reader(8888 + i, bind_hostname="::1")

    def prepare_senders(self, thread_pool, n):
        if n == 1:
            with pytest.deprecated_call():
                return spead2.send.UdpStream(
                    thread_pool, "::1", 8888,
                    spead2.send.StreamConfig(rate=1e7),
                    buffer_size=0)
        else:
            return spead2.send.UdpStream(
                thread_pool,
                [("::1", 8888 + i) for i in range(n)],
                spead2.send.StreamConfig(rate=1e7),
                buffer_size=0)


class TestPassthroughUdpCustomSocket(BaseTestPassthroughSubstreams):
    is_lossy = True

    def prepare_receivers(self, receivers):
        self._ports = []
        for i, receiver in enumerate(receivers):
            recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            recv_sock.bind(("127.0.0.1", 0))
            self._ports.append(recv_sock.getsockname()[1])
            receiver.add_udp_reader(socket=recv_sock)
            recv_sock.close()   # spead2 duplicates the socket

    def prepare_senders(self, thread_pool, n):
        assert len(self._ports) == n
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if n == 1:
            with pytest.deprecated_call():
                sender = spead2.send.UdpStream(
                    thread_pool, send_sock, "127.0.0.1", self._ports[0],
                    spead2.send.StreamConfig(rate=1e7))
        else:
            sender = spead2.send.UdpStream(
                thread_pool, send_sock,
                [("127.0.0.1", port) for port in self._ports],
                spead2.send.StreamConfig(rate=1e7))
        send_sock.close()   # spead2 duplicates the socket
        return sender


class TestPassthroughUdpMulticast(BaseTestPassthroughSubstreams):
    is_lossy = True
    MCAST_GROUP = '239.255.88.88'
    INTERFACE_ADDRESS = '127.0.0.1'

    def prepare_receivers(self, receivers):
        for i, receiver in enumerate(receivers):
            receiver.add_udp_reader(
                self.MCAST_GROUP, 8887 - i, interface_address=self.INTERFACE_ADDRESS)

    def prepare_senders(self, thread_pool, n):
        if n == 1:
            with pytest.deprecated_call():
                return spead2.send.UdpStream(
                    thread_pool, self.MCAST_GROUP, 8887,
                    spead2.send.StreamConfig(rate=1e7),
                    buffer_size=0, ttl=1, interface_address=self.INTERFACE_ADDRESS)
        else:
            return spead2.send.UdpStream(
                thread_pool,
                [(self.MCAST_GROUP, 8887 - i) for i in range(n)],
                spead2.send.StreamConfig(rate=1e7),
                buffer_size=0, ttl=1, interface_address=self.INTERFACE_ADDRESS)


class TestPassthroughUdp6Multicast(TestPassthroughUdp6):
    MCAST_GROUP = 'ff14::1234'

    @classmethod
    def get_interface_index(cls):
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET6, [])
            for addr in addrs:
                if addr['addr'] != '::1':
                    return socket.if_nametoindex(iface)
        pytest.skip('could not find suitable interface for test')

    def prepare_receivers(self, receivers):
        interface_index = self.get_interface_index()
        for i, receiver in enumerate(receivers):
            receiver.add_udp_reader(self.MCAST_GROUP, 8887 - i, interface_index=interface_index)

    def prepare_senders(self, thread_pool, n):
        interface_index = self.get_interface_index()
        if n == 1:
            with pytest.deprecated_call():
                return spead2.send.UdpStream(
                    thread_pool, self.MCAST_GROUP, 8887,
                    spead2.send.StreamConfig(rate=1e7),
                    buffer_size=0, ttl=0, interface_index=interface_index)
        else:
            return spead2.send.UdpStream(
                thread_pool,
                [(self.MCAST_GROUP, 8887 - i) for i in range(n)],
                spead2.send.StreamConfig(rate=1e7),
                buffer_size=0, ttl=0, interface_index=interface_index)


class TestPassthroughUdpIbv(BaseTestPassthroughSubstreams):
    is_lossy = True
    MCAST_GROUP = '239.255.88.88'

    def _interface_address(self):
        ifaddr = os.getenv('SPEAD2_TEST_IBV_INTERFACE_ADDRESS')
        if not ifaddr:
            pytest.skip('Envar SPEAD2_TEST_IBV_INTERFACE_ADDRESS not set')
        return ifaddr

    def setup(self):
        # mlx5 drivers only enable multicast loopback if there are multiple
        # device contexts. The sender and receiver end up sharing one, so we
        # need to explicitly create another.
        if not hasattr(spead2, 'IbvContext'):
            pytest.skip('IBV support not compiled in')
        self._extra_context = spead2.IbvContext(self._interface_address())

    def teardown(self):
        self._extra_context.reset()

    def prepare_receivers(self, receivers):
        for i, receiver in enumerate(receivers):
            receiver.add_udp_ibv_reader(
                spead2.recv.UdpIbvConfig(
                    endpoints=[(self.MCAST_GROUP, 8876 + i)],
                    interface_address=self._interface_address()))

    def prepare_senders(self, thread_pool, n):
        # The buffer size is deliberately reduced so that we test the
        # wrapping once the buffer has been used up
        if n == 1:
            with pytest.deprecated_call():
                return spead2.send.UdpIbvStream(
                    thread_pool, self.MCAST_GROUP, 8876,
                    spead2.send.StreamConfig(rate=1e7),
                    self._interface_address(),
                    buffer_size=64 * 1024)
        else:
            return spead2.send.UdpIbvStream(
                thread_pool,
                spead2.send.StreamConfig(rate=1e7),
                spead2.send.UdpIbvConfig(
                    endpoints=[(self.MCAST_GROUP, 8876 + i) for i in range(n)],
                    interface_address=self._interface_address(),
                    buffer_size=64 * 1024
                )
            )

    @pytest.mark.parametrize('num_items', [0, 1, 3, 4, 10])
    def test_memory_regions(self, num_items):
        receiver = spead2.recv.Stream(spead2.ThreadPool(), spead2.recv.StreamConfig())
        receiver.add_udp_ibv_reader(
            spead2.recv.UdpIbvConfig(
                endpoints=[(self.MCAST_GROUP, 8876)],
                interface_address=self._interface_address()))

        ig = spead2.send.ItemGroup()
        data = [np.random.randn(50) for i in range(num_items)]
        for i in range(num_items):
            ig.add_item(id=0x2345 + i, name=f'name {i}', description=f'description {i}',
                        shape=data[i].shape, dtype=data[i].dtype, value=data[i])
        sender = spead2.send.UdpIbvStream(
            spead2.ThreadPool(),
            spead2.send.StreamConfig(rate=1e7),
            spead2.send.UdpIbvConfig(
                endpoints=[(self.MCAST_GROUP, 8876)],
                interface_address=self._interface_address(),
                memory_regions=data
            )
        )
        sender.send_heap(ig.get_heap())
        sender.send_heap(ig.get_end())

        recv_ig = spead2.ItemGroup()
        for heap in receiver:
            recv_ig.update(heap)
        assert_item_groups_equal(ig, recv_ig)


class TestPassthroughTcp(BaseTestPassthrough):
    def prepare_receiver(self, receiver):
        receiver.add_tcp_reader(8887, bind_hostname="127.0.0.1")

    def prepare_sender(self, thread_pool):
        return spead2.send.TcpStream(thread_pool, [("127.0.0.1", 8887)])


class TestPassthroughTcpCustomSocket(BaseTestPassthrough):
    def prepare_receiver(self, receiver):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self._port = sock.getsockname()[1]
        sock.listen(1)
        receiver.add_tcp_reader(sock)

    def prepare_sender(self, thread_pool):
        sock = socket.socket()
        sock.connect(("127.0.0.1", self._port))
        return spead2.send.TcpStream(thread_pool, sock)


class TestPassthroughTcp6(BaseTestPassthrough):
    requires_ipv6 = True

    def prepare_receiver(self, receiver):
        receiver.add_tcp_reader(8887, bind_hostname="::1")

    def prepare_sender(self, thread_pool):
        with pytest.deprecated_call():
            return spead2.send.TcpStream(thread_pool, "::1", 8887)


class TestPassthroughMem(BaseTestPassthrough):
    def transmit_item_groups(self, item_groups, *,
                             memcpy, allocator, new_order='=', round_robin=False):
        assert len(item_groups) == 1
        assert not round_robin
        thread_pool = spead2.ThreadPool(2)
        sender = spead2.send.BytesStream(thread_pool)
        gen = spead2.send.HeapGenerator(item_groups[0])
        sender.send_heap(gen.get_heap())
        sender.send_heap(gen.get_end())
        recv_config = spead2.recv.StreamConfig(memcpy=memcpy)
        if allocator is not None:
            recv_config.memory_allocator = allocator
        receiver = spead2.recv.Stream(thread_pool, recv_config)
        receiver.add_buffer_reader(sender.getvalue())
        received_item_group = spead2.ItemGroup()
        for heap in receiver:
            received_item_group.update(heap, new_order)
        return [received_item_group]


class TestPassthroughInproc(BaseTestPassthroughSubstreams):
    def prepare_receivers(self, receivers):
        assert len(receivers) == len(self._queues)
        for receiver, queue in zip(receivers, self._queues):
            receiver.add_inproc_reader(queue)

    def prepare_senders(self, thread_pool, n):
        assert n == len(self._queues)
        if n == 1:
            with pytest.deprecated_call():
                return spead2.send.InprocStream(thread_pool, self._queues[0])
        else:
            return spead2.send.InprocStream(thread_pool, self._queues)

    def transmit_item_groups(self, item_groups, *,
                             memcpy, allocator, new_order='=', round_robin=False):
        self._queues = [spead2.InprocQueue() for ig in item_groups]
        ret = super().transmit_item_groups(
            item_groups, memcpy=memcpy, allocator=allocator,
            new_order=new_order, round_robin=round_robin)
        for queue in self._queues:
            queue.stop()
        return ret

    def test_queues(self):
        queues = [spead2.InprocQueue() for i in range(2)]
        stream = spead2.send.InprocStream(spead2.ThreadPool(), queues)
        assert stream.queues == queues
        with pytest.deprecated_call():
            with pytest.raises(RuntimeError):
                stream.queue

    def test_queue(self):
        queue = spead2.InprocQueue()
        stream = spead2.send.InprocStream(spead2.ThreadPool(), [queue])
        assert stream.queues == [queue]
        with pytest.deprecated_call():
            assert stream.queue is queue
