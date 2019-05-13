# Copyright 2015 SKA South Africa
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

"""Test that data can be passed over the SPEAD protocol, using the various
transports, and mixing with the old PySPEAD implementation.
"""

from __future__ import division, print_function
import os
import io
import socket
import sys

import numpy as np
import netifaces
from decorator import decorator
from nose.tools import assert_equal, timed
from nose.plugins.skip import SkipTest
try:
    import spead64_40
except ImportError:
    spead64_40 = None
try:
    import spead64_48
except ImportError:
    spead64_48 = None
try:
    from socket import if_nametoindex
except ImportError:
    import ctypes
    import ctypes.util
    _libc_name = ctypes.util.find_library('c')
    if _libc_name is None:
        raise
    _libc = ctypes.CDLL(_libc_name, use_errno=True)

    def if_nametoindex(name):       # type: ignore
        if not isinstance(name, bytes):
            name = name.encode('utf-8')
        ret = _libc.if_nametoindex(name)
        if ret == 0:
            raise OSError(ctypes.get_errno(), 'if_nametoindex failed')
        else:
            return ret

import spead2
import spead2.send
import spead2.recv


def _assert_items_equal(item1, item2):
    assert_equal(item1.id, item2.id)
    assert_equal(item1.name, item2.name)
    assert_equal(item1.description, item2.description)
    assert_equal(item1.shape, item2.shape)
    assert_equal(item1.format, item2.format)
    # Byte order need not match, provided that values are received correctly
    if item1.dtype is not None and item2.dtype is not None:
        assert_equal(item1.dtype.newbyteorder('<'), item2.dtype.newbyteorder('<'))
    else:
        assert_equal(item1.dtype, item2.dtype)
    assert_equal(item1.order, item2.order)
    # Comparing arrays has many issues. Convert them to lists where appropriate
    value1 = item1.value
    value2 = item2.value
    if hasattr(value1, 'tolist'):
        value1 = value1.tolist()
    if hasattr(value2, 'tolist'):
        value2 = value2.tolist()
    assert_equal(value1, value2)


def assert_item_groups_equal(item_group1, item_group2):
    assert_equal(sorted(item_group1.keys()), sorted(item_group2.keys()))
    for key in item_group1.keys():
        _assert_items_equal(item_group1[key], item_group2[key])


@decorator
def no_legacy_send(test, *args, **kwargs):
    if not args[0].is_legacy_send:
        test(*args, **kwargs)


@decorator
def no_legacy_receive(test, *args, **kwargs):
    if not args[0].is_legacy_receive:
        test(*args, **kwargs)


@decorator
def no_legacy(test, *args, **kwargs):
    if not (args[0].is_legacy_send or args[0].is_legacy_receive):
        test(*args, **kwargs)


def timed_class(cls):
    """Class decorator version of `nose.tools.timed`"""
    for key in cls.__dict__:
        if key.startswith('test_'):
            setattr(cls, key, timed(2)(getattr(cls, key)))
    return cls


@timed_class
class BaseTestPassthrough(object):
    """Tests common to all transports and libraries"""

    is_legacy_send = False
    is_legacy_receive = False
    is_lossy = False

    def _test_item_group(self, item_group, memcpy=spead2.MEMCPY_STD, allocator=None):
        received_item_group = self.transmit_item_group(item_group, memcpy, allocator)
        assert_item_groups_equal(item_group, received_item_group)
        if not self.is_legacy_receive:
            for item in received_item_group.values():
                if item.dtype is not None:
                    assert_equal(item.value.dtype, item.value.dtype.newbyteorder('='))

    def test_numpy_simple(self):
        """A basic array with numpy encoding"""
        ig = spead2.send.ItemGroup()
        data = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.uint16)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=data.shape, dtype=data.dtype, value=data)
        self._test_item_group(ig)

    def test_numpy_large(self):
        """A numpy style array split across several packets. It also
        uses non-temporal copies, a custom allocator, and a memory pool,
        to test that those all work."""
        # macOS doesn't have a big enough socket buffer to reliably transmit
        # the whole thing over UDP.
        if self.is_lossy and sys.platform == 'darwin':
            raise SkipTest("macOS can't reliably handle large heaps over UDP")
        ig = spead2.send.ItemGroup()
        data = np.random.randn(100, 200)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=data.shape, dtype=data.dtype, value=data)
        allocator = spead2.MmapAllocator()
        pool = spead2.MemoryPool(1, 4096, 4, 4, allocator)
        self._test_item_group(ig, spead2.MEMCPY_NONTEMPORAL, pool)

    def test_fallback_struct_whole_bytes(self):
        """A structure with non-byte-aligned elements, but which is
        byte-aligned overall."""
        ig = spead2.send.ItemGroup()
        format = [('u', 4), ('f', 64), ('i', 4)]
        data = (12, 1.5, -3)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(), format=format, value=data)
        self._test_item_group(ig)

    @no_legacy_receive
    def test_string(self):
        """Byte string is converted to array of characters and back.

        It is disabled for PySPEAD receive because PySPEAD requires a
        non-standard 's' conversion to do this correctly.
        """
        ig = spead2.send.ItemGroup()
        format = [('c', 8)]
        data = 'Hello world'
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(None,), format=format, value=data)
        self._test_item_group(ig)

    @no_legacy_receive
    def test_fallback_array_partial_bytes_small(self):
        """An array which takes a fractional number of bytes per element
        and is small enough to encode in an immediate.

        It is disabled for PySPEAD receive because PySPEAD does not decode
        such items in the same way as it encodes them.
        """
        ig = spead2.send.ItemGroup()
        format = [('u', 7)]
        data = [127, 12, 123]
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(len(data),), format=format, value=data)
        self._test_item_group(ig)

    @no_legacy
    def test_fallback_types(self):
        """An array structure using a mix of types."""
        ig = spead2.send.ItemGroup()
        format = [('b', 1), ('i', 7), ('c', 8), ('f', 32)]
        data = [(True, 17, b'y', 1.0), (False, -23, b'n', -1.0)]
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(2,), format=format, value=data)
        self._test_item_group(ig)

    @no_legacy
    def test_numpy_fallback_struct(self):
        """A structure specified using a format, but which is encodable using
        numpy."""
        ig = spead2.send.ItemGroup()
        format = [('u', 8), ('f', 32)]
        data = (12, 1.5)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(), format=format, value=data)
        self._test_item_group(ig)

    @no_legacy
    def test_fallback_struct_partial_bytes(self):
        """A structure which takes a fractional number of bytes per element.
        """
        ig = spead2.send.ItemGroup()
        format = [('u', 4), ('f', 64)]
        data = (12, 1.5)
        ig.add_item(id=0x2345, name='name', description='description',
                    shape=(), format=format, value=data)
        self._test_item_group(ig)

    def test_fallback_scalar(self):
        """Send a scalar using fallback format descriptor.

        PySPEAD has a bug that makes this fail if the format is upgraded to a
        numpy descriptor.
        """
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
            name = 'test item {}'.format(i)
            ig.add_item(id=0x2345 + i, name=name, description=name,
                        shape=(), format=[('u', 40)], value=0x12345 * i)
        self._test_item_group(ig)

    def transmit_item_group(self, item_group, memcpy, allocator):
        """Transmit `item_group` over the chosen transport,
        and return the item group received at the other end. Subclasses
        should override this.
        """
        thread_pool = spead2.ThreadPool(2)
        receiver = spead2.recv.Stream(thread_pool)
        receiver.set_memcpy(memcpy)
        if allocator is not None:
            receiver.set_memory_allocator(allocator)
        self.prepare_receiver(receiver)
        sender = self.prepare_sender(thread_pool)
        gen = spead2.send.HeapGenerator(item_group)
        sender.send_heap(gen.get_heap())
        sender.send_heap(gen.get_end())
        received_item_group = spead2.ItemGroup()
        for heap in receiver:
            received_item_group.update(heap)
        return received_item_group

    def prepare_receiver(self, receiver):
        """Generate a receiver to use in the test"""
        raise NotImplementedError()

    def prepare_sender(self, thread_pool):
        """Generate a sender to use in the test"""
        raise NotImplementedError()


class BaseTestPassthroughIPv6(BaseTestPassthrough):
    @classmethod
    def check_ipv6(cls):
        if not socket.has_ipv6:
            raise SkipTest('platform does not support IPv6')
        # Travis' Trusty image fails to bind to an IPv6 address
        sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        try:
            sock.bind(("::1", 8888))
        except IOError:
            raise SkipTest('platform cannot bind IPv6 localhost address')
        finally:
            sock.close()

    def transmit_item_group(self, item_group, memcpy, allocator):
        self.check_ipv6()
        return super(BaseTestPassthroughIPv6, self).transmit_item_group(
            item_group, memcpy, allocator)


class TestPassthroughUdp(BaseTestPassthrough):
    is_lossy = True

    def prepare_receiver(self, receiver):
        receiver.add_udp_reader(8888, bind_hostname="localhost")

    def prepare_sender(self, thread_pool):
        return spead2.send.UdpStream(
            thread_pool, "localhost", 8888,
            spead2.send.StreamConfig(rate=1e7),
            buffer_size=0)


class TestPassthroughUdp6(BaseTestPassthroughIPv6):
    is_lossy = True

    def prepare_receiver(self, receiver):
        receiver.add_udp_reader(8888, bind_hostname="::1")

    def prepare_sender(self, thread_pool):
        return spead2.send.UdpStream(
            thread_pool, "::1", 8888,
            spead2.send.StreamConfig(rate=1e7),
            buffer_size=0)


class TestPassthroughUdpCustomSocket(BaseTestPassthrough):
    is_lossy = True

    def prepare_receiver(self, receiver):
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", 0))
        self._port = recv_sock.getsockname()[1]
        receiver.add_udp_reader(socket=recv_sock)
        recv_sock.close()   # spead2 duplicates the socket

    def prepare_sender(self, thread_pool):
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender = spead2.send.UdpStream(
            thread_pool, send_sock, "127.0.0.1", self._port,
            spead2.send.StreamConfig(rate=1e7))
        send_sock.close()   # spead2 duplicates the socket
        return sender


class TestPassthroughUdpMulticast(BaseTestPassthrough):
    is_lossy = True
    MCAST_GROUP = '239.255.88.88'
    INTERFACE_ADDRESS = '127.0.0.1'

    def prepare_receiver(self, receiver):
        receiver.add_udp_reader(
            self.MCAST_GROUP, 8887, interface_address=self.INTERFACE_ADDRESS)

    def prepare_sender(self, thread_pool):
        return spead2.send.UdpStream(
            thread_pool, self.MCAST_GROUP, 8887,
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
                    return if_nametoindex(iface)
        raise SkipTest('could not find suitable interface for test')

    def prepare_receiver(self, receiver):
        interface_index = self.get_interface_index()
        receiver.add_udp_reader(self.MCAST_GROUP, 8887, interface_index=interface_index)

    def prepare_sender(self, thread_pool):
        interface_index = self.get_interface_index()
        return spead2.send.UdpStream(
            thread_pool, self.MCAST_GROUP, 8887,
            spead2.send.StreamConfig(rate=1e7),
            buffer_size=0, ttl=0, interface_index=interface_index)


class TestPassthroughUdpIbv(BaseTestPassthrough):
    is_lossy = True
    MCAST_GROUP = '239.255.88.88'

    def _interface_address(self):
        ifaddr = os.getenv('SPEAD2_TEST_IBV_INTERFACE_ADDRESS')
        if not ifaddr:
            raise SkipTest('Envar SPEAD2_TEST_IBV_INTERFACE_ADDRESS not set')
        return ifaddr

    def setup(self):
        # mlx5 drivers only enable multicast loopback if there are multiple
        # device contexts. The sender and receiver end up sharing one, so we
        # need to explicitly create another.
        if not hasattr(spead2, 'IbvContext'):
            raise SkipTest('IBV support not compiled in')
        self._extra_context = spead2.IbvContext(self._interface_address())

    def teardown(self):
        self._extra_context.reset()

    def prepare_receiver(self, receiver):
        receiver.add_udp_ibv_reader([(self.MCAST_GROUP, 8886)], self._interface_address())

    def prepare_sender(self, thread_pool):
        return spead2.send.UdpIbvStream(
            thread_pool, self.MCAST_GROUP, 8886,
            spead2.send.StreamConfig(rate=1e7),
            self._interface_address())


class TestPassthroughTcp(BaseTestPassthrough):
    def prepare_receiver(self, receiver):
        receiver.add_tcp_reader(8887, bind_hostname="127.0.0.1")

    def prepare_sender(self, thread_pool):
        return spead2.send.TcpStream(thread_pool, "127.0.0.1", 8887)


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


class TestPassthroughTcp6(BaseTestPassthroughIPv6):
    def prepare_receiver(self, receiver):
        receiver.add_tcp_reader(8887, bind_hostname="::1")

    def prepare_sender(self, thread_pool):
        return spead2.send.TcpStream(thread_pool, "::1", 8887)


class TestPassthroughMem(BaseTestPassthrough):
    def transmit_item_group(self, item_group, memcpy, allocator):
        thread_pool = spead2.ThreadPool(2)
        sender = spead2.send.BytesStream(thread_pool)
        gen = spead2.send.HeapGenerator(item_group)
        sender.send_heap(gen.get_heap())
        sender.send_heap(gen.get_end())
        receiver = spead2.recv.Stream(thread_pool)
        receiver.set_memcpy(memcpy)
        if allocator is not None:
            receiver.set_memory_allocator(allocator)
        receiver.add_buffer_reader(sender.getvalue())
        received_item_group = spead2.ItemGroup()
        for heap in receiver:
            received_item_group.update(heap)
        return received_item_group


class TestPassthroughInproc(BaseTestPassthrough):
    def prepare_receiver(self, receiver):
        receiver.add_inproc_reader(self._queue)

    def prepare_sender(self, thread_pool):
        return spead2.send.InprocStream(thread_pool, self._queue)

    def transmit_item_group(self, item_group, memcpy, allocator):
        self._queue = spead2.InprocQueue()
        ret = super(TestPassthroughInproc, self).transmit_item_group(item_group, memcpy, allocator)
        self._queue.stop()
        return ret


class TestAllocators(BaseTestPassthrough):
    """Like TestPassthroughMem, but uses some custom allocators"""
    def transmit_item_group(self, item_group, memcpy, allocator):
        thread_pool = spead2.ThreadPool(2)
        sender = spead2.send.BytesStream(thread_pool)
        gen = spead2.send.HeapGenerator(item_group)
        sender.send_heap(gen.get_heap())
        sender.send_heap(gen.get_end())
        receiver = spead2.recv.Stream(thread_pool)
        if allocator is not None:
            receiver.set_memory_allocator(allocator)
        receiver.set_memcpy(memcpy)
        receiver.add_buffer_reader(sender.getvalue())
        received_item_group = spead2.ItemGroup()
        for heap in receiver:
            received_item_group.update(heap)
        return received_item_group


class BaseTestPassthroughLegacySend(BaseTestPassthrough):
    is_legacy_send = True

    def transmit_item_group(self, item_group, memcpy, allocator):
        if not self.spead:
            raise SkipTest('spead module not importable')
        transport = io.BytesIO()
        sender = self.spead.Transmitter(transport)
        legacy_item_group = self.spead.ItemGroup()
        for item in item_group.values():
            # PySPEAD only supports either 1D variable or fixed-size
            if item.is_variable_size():
                assert len(item.shape) == 1
                shape = -1
            else:
                shape = item.shape
            legacy_item_group.add_item(
                id=item.id,
                name=item.name,
                description=item.description,
                shape=shape,
                fmt=self.spead.mkfmt(*item.format) if item.format else self.spead.DEFAULT_FMT,
                ndarray=np.array(item.value) if not item.format else None)
            legacy_item_group[item.name] = item.value
        sender.send_heap(legacy_item_group.get_heap())
        sender.end()
        thread_pool = spead2.ThreadPool(1)
        receiver = spead2.recv.Stream(thread_pool, bug_compat=spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        receiver.set_memcpy(memcpy)
        if allocator is not None:
            receiver.set_memory_allocator(allocator)
        receiver.add_buffer_reader(transport.getvalue())
        received_item_group = spead2.ItemGroup()
        for heap in receiver:
            received_item_group.update(heap)
        return received_item_group


class TestPassthroughLegacySend64_40(BaseTestPassthroughLegacySend):
    spead = spead64_40


class TestPassthroughLegacySend64_48(BaseTestPassthroughLegacySend):
    spead = spead64_48


class BaseTestPassthroughLegacyReceive(BaseTestPassthrough):
    is_legacy_receive = True

    def transmit_item_group(self, item_group, memcpy, allocator):
        if not self.spead:
            raise SkipTest('spead module not importable')
        thread_pool = spead2.ThreadPool(1)
        sender = spead2.send.BytesStream(thread_pool)
        gen = spead2.send.HeapGenerator(item_group, flavour=self.flavour)
        sender.send_heap(gen.get_heap())
        sender.send_heap(gen.get_end())
        receiver = self.spead.TransportString(sender.getvalue())
        legacy_item_group = self.spead.ItemGroup()
        for heap in self.spead.iterheaps(receiver):
            legacy_item_group.update(heap)
        received_item_group = spead2.ItemGroup()
        for key in legacy_item_group.keys():
            item = legacy_item_group.get_item(key)
            # PySPEAD indicates 1D variable as -1 (scalar), everything else is fixed-sized
            if item.shape == -1:
                shape = (None,)
            else:
                shape = item.shape
            if item.dtype is None:
                received_item_group.add_item(
                    id=item.id,
                    name=item.name,
                    description=item.description,
                    shape=shape,
                    format=list(self.spead.parsefmt(item.format)),
                    value=item.get_value())
            else:
                received_item_group.add_item(
                    id=item.id,
                    name=item.name,
                    description=item.description,
                    shape=shape,
                    dtype=item.dtype,
                    order='F' if item.fortran_order else 'C',
                    value=item.get_value())
        return received_item_group


class TestPassthroughLegacyReceive64_40(BaseTestPassthroughLegacyReceive):
    spead = spead64_40
    flavour = spead2.Flavour(4, 64, 40, spead2.BUG_COMPAT_PYSPEAD_0_5_2)


class TestPassthroughLegacyReceive64_48(BaseTestPassthroughLegacyReceive):
    spead = spead64_48
    flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
