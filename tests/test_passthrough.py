# Copyright 2015, 2019-2024 National Research Foundation (SARAO)
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

import asyncio
import ipaddress
import os
import socket
import sys

import netifaces
import numpy as np
import pytest

import spead2
import spead2.recv.asyncio
import spead2.send.asyncio


def _assert_items_equal(item1, item2):
    assert item1.id == item2.id
    assert item1.name == item2.name
    assert item1.description == item2.description
    assert item1.shape == item2.shape
    assert item1.format == item2.format
    # Byte order need not match, provided that values are received correctly
    if item1.dtype is not None and item2.dtype is not None:
        assert item1.dtype.newbyteorder("<") == item2.dtype.newbyteorder("<")
    else:
        assert item1.dtype == item2.dtype
    assert item1.order == item2.order
    # Comparing arrays has many issues. Convert them to lists where appropriate
    value1 = item1.value
    value2 = item2.value
    if hasattr(value1, "tolist"):
        value1 = value1.tolist()
    if hasattr(value2, "tolist"):
        value2 = value2.tolist()
    assert value1 == value2


def assert_item_groups_equal(item_group1, item_group2):
    assert sorted(item_group1.keys()) == sorted(item_group2.keys())
    for key in item_group1.keys():
        _assert_items_equal(item_group1[key], item_group2[key])


class Transport:
    """Encapsulates a transport mechanism.

    Each test will instantiate an instance of a subclass and use it to
    configure senders and receivers.
    """

    is_lossy = False
    supports_substreams = True
    requires_ipv6 = False
    requires_ipv4_multicast = False
    requires_ipv6_multicast = False

    def __init__(self) -> None:
        self._udp_ports: set[int] = set()
        self._tcp_ports: set[int] = set()

    def _unused_port(self, socket_type: int, used: set[int]) -> int:
        """Obtain an available port of the specified type.

        It is guaranteed to be distinct from any ports in `used.` On
        return, the port is added to `used`.
        """
        # This implementation is loosely based on pytest-asyncio's
        # unused_tcp_port_factory fixture.
        while True:
            with socket.socket(type=socket_type) as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
                if port not in used:
                    used.add(port)
                    return port

    def unused_udp_port(self):
        return self._unused_port(socket.SOCK_DGRAM, self._udp_ports)

    def unused_tcp_port(self):
        return self._unused_port(socket.SOCK_STREAM, self._tcp_ports)

    @classmethod
    def check_platform(cls):
        if cls.requires_ipv6:
            if not socket.has_ipv6:
                pytest.skip("platform does not support IPv6")
            with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as sock:
                # Travis build systems fail to bind to an IPv6 address
                try:
                    sock.bind(("::1", 0))
                except OSError:
                    pytest.skip("platform cannot bind IPv6 localhost address")
        if cls.requires_ipv4_multicast:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                # qemu doesn't yet support IP_MULTICAST_IF socket option
                # (https://gitlab.com/qemu-project/qemu/-/issues/1837)
                # Skip the test if it is non-functional.
                try:
                    loopback = ipaddress.ip_address("127.0.0.1")
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, loopback.packed)
                except OSError:
                    pytest.skip("platform cannot set multicast interface (might be qemu?)")
        if cls.requires_ipv6_multicast:
            with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as sock:
                # Github Actions on MacOS doesn't have routes to multicast
                try:
                    sock.connect(("ff14::1234", 0))
                    sock.send(b"test")
                except OSError:
                    pytest.skip("platform cannot transmit to an IPv6 multicast address")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @classmethod
    def transmit_item_groups(
        cls,
        item_groups,
        *,
        memcpy,
        allocator,
        explicit_start,
        new_order="=",
        group_mode=None,
    ):
        raise NotImplementedError()


class SyncTransport(Transport):
    def prepare_receivers(self, receivers):
        """Add readers to receivers.

        The return value will be passed to :meth:`prepare_senders`.
        """
        raise NotImplementedError()

    def prepare_senders(self, thread_pool, n, receiver_state):
        """Generate a sender to use in the test, with `n` substreams."""
        raise NotImplementedError()

    @classmethod
    def transmit_item_groups(
        cls,
        item_groups,
        *,
        memcpy,
        allocator,
        explicit_start,
        new_order="=",
        group_mode=None,
    ):
        """Transmit `item_groups` over the chosen transport.

        Return the item groups received at the other end. Each item group will
        be transmitted over a separate substream (thus, the transport must
        support substreams if `item_groups` has more than one element).
        """
        cls.check_platform()
        with cls() as transport:
            recv_config = spead2.recv.StreamConfig(memcpy=memcpy, explicit_start=explicit_start)
            if allocator is not None:
                recv_config.memory_allocator = allocator
            receivers = [
                spead2.recv.Stream(spead2.ThreadPool(), recv_config)
                for i in range(len(item_groups))
            ]
            receiver_state = transport.prepare_receivers(receivers)
            if explicit_start:
                for receiver in receivers:
                    receiver.start()
            sender = transport.prepare_senders(
                spead2.ThreadPool(), len(item_groups), receiver_state
            )
            gens = [spead2.send.HeapGenerator(item_group) for item_group in item_groups]
            if len(item_groups) != 1:
                # Use reversed order so that if everything is actually going
                # through the same transport it will get picked up.
                if group_mode is not None:
                    sender.send_heaps(
                        [
                            spead2.send.HeapReference(gen.get_heap(), substream_index=i)
                            for i, gen in reversed(list(enumerate(gens)))
                        ],
                        group_mode,
                    )
                    # For the stop heaps, use a HeapReferenceList to test it.
                    hrl = spead2.send.HeapReferenceList(
                        [
                            spead2.send.HeapReference(gen.get_end(), substream_index=i)
                            for i, gen in enumerate(gens)
                        ]
                    )
                    # Also test splitting the list
                    sender.send_heaps(hrl[:1], group_mode)
                    sender.send_heaps(hrl[1:], group_mode)
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


class AsyncTransport(Transport):
    @classmethod
    async def transmit_item_groups_async(
        cls, item_groups, *, memcpy, allocator, explicit_start, new_order="=", group_mode=None
    ):
        cls.check_platform()
        with cls() as transport:
            recv_config = spead2.recv.StreamConfig(memcpy=memcpy, explicit_start=explicit_start)
            if allocator is not None:
                recv_config.memory_allocator = allocator
            receivers = [
                spead2.recv.asyncio.Stream(spead2.ThreadPool(), recv_config)
                for i in range(len(item_groups))
            ]
            receiver_state = await transport.prepare_receivers(receivers)
            if explicit_start:
                for receiver in receivers:
                    receiver.start()
            sender = await transport.prepare_senders(
                spead2.ThreadPool(), len(item_groups), receiver_state
            )
            gens = [spead2.send.HeapGenerator(item_group) for item_group in item_groups]
            if len(item_groups) != 1:
                # Use reversed order so that if everything is actually going
                # through the same transport it will get picked up.
                if group_mode is not None:
                    await sender.async_send_heaps(
                        [
                            spead2.send.HeapReference(gen.get_heap(), substream_index=i)
                            for i, gen in reversed(list(enumerate(gens)))
                        ],
                        group_mode,
                    )
                    # Use a HeapReferenceList to test it
                    hrl = spead2.send.HeapReferenceList(
                        [
                            spead2.send.HeapReference(gen.get_end(), substream_index=i)
                            for i, gen in enumerate(gens)
                        ]
                    )
                    # Also test splitting the list
                    await sender.async_send_heaps(hrl[:1], group_mode)
                    await sender.async_send_heaps(hrl[1:], group_mode)
                else:
                    for i, gen in reversed(list(enumerate(gens))):
                        await sender.async_send_heap(gen.get_heap(), substream_index=i)
                    for i, gen in enumerate(gens):
                        await sender.async_send_heap(gen.get_end(), substream_index=i)
            else:
                # This is a separate code path to give coverage of the case where
                # the substream index is implicit.
                await sender.async_send_heap(gens[0].get_heap())
                await sender.async_send_heap(gens[0].get_end())
            await sender.async_flush()
            received_item_groups = []
            for receiver in receivers:
                ig = spead2.ItemGroup()
                for heap in receiver:
                    ig.update(heap, new_order)
                received_item_groups.append(ig)
            return received_item_groups

    @classmethod
    def transmit_item_groups(
        cls,
        item_groups,
        *,
        memcpy,
        allocator,
        explicit_start,
        new_order="=",
        group_mode=None,
    ):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                cls.transmit_item_groups_async(
                    item_groups,
                    memcpy=memcpy,
                    allocator=allocator,
                    explicit_start=explicit_start,
                    new_order=new_order,
                    group_mode=group_mode,
                )
            )
        finally:
            loop.close()


class SyncNoSubstreamsTransport(SyncTransport):
    supports_substreams = False

    def prepare_receiver(self, receiver):
        """Add a reader to a single receiver.

        The return value will be passed to :meth:`prepare_sender`.
        """
        raise NotImplementedError()

    def prepare_sender(self, thread_pool, receiver_state):
        """Generate a sender with a single substream."""
        raise NotImplementedError()

    def prepare_receivers(self, receivers):
        assert len(receivers) == 1
        return self.prepare_receiver(receivers[0])

    def prepare_senders(self, thread_pool, n, receiver_state):
        """Generate a sender to use in the test, with `n` substreams."""
        assert n == 1
        return self.prepare_sender(thread_pool, receiver_state)


class AsyncNoSubstreamsTransport(AsyncTransport):
    supports_substreams = False

    async def prepare_receiver(self, receiver):
        """Add a reader to a single receiver.

        The return value will be passed to :meth:`prepare_sender`.
        """
        raise NotImplementedError()

    async def prepare_sender(self, thread_pool, receiver_state):
        """Generate a sender with a single substream."""
        raise NotImplementedError()

    async def prepare_receivers(self, receivers):
        assert len(receivers) == 1
        return await self.prepare_receiver(receivers[0])

    async def prepare_senders(self, thread_pool, n, receiver_state):
        """Generate a sender to use in the test, with `n` substreams."""
        assert n == 1
        return await self.prepare_sender(thread_pool, receiver_state)


class UdpTransport(SyncTransport):
    is_lossy = True

    def prepare_receivers(self, receivers, bind_hostname="localhost"):
        ports = []
        for receiver in receivers:
            port = self.unused_udp_port()
            receiver.add_udp_reader(port, bind_hostname=bind_hostname)
            ports.append(port)
        return [(bind_hostname, port) for port in ports]

    def prepare_senders(self, thread_pool, n, endpoints, cls=spead2.send.UdpStream):
        assert len(endpoints) == n
        return cls(
            thread_pool,
            endpoints,
            spead2.send.StreamConfig(rate=1e7),
            buffer_size=0,
        )


class AsyncUdpTransport(AsyncTransport):
    is_lossy = True

    async def prepare_receivers(self, receivers):
        return UdpTransport.prepare_receivers(self, receivers)

    async def prepare_senders(self, thread_pool, n, ports):
        return UdpTransport.prepare_senders(
            self, thread_pool, n, ports, cls=spead2.send.asyncio.UdpStream
        )


class Udp6Transport(UdpTransport):
    is_lossy = True
    requires_ipv6 = True

    def prepare_receivers(self, receivers):
        return super().prepare_receivers(receivers, bind_hostname="::1")


class UdpCustomSocketTransport(SyncTransport):
    is_lossy = True

    def prepare_receivers(self, receivers):
        ports = []
        for i, receiver in enumerate(receivers):
            # spead2 duplicates the socket, so we use a context manager to
            # close the original.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as recv_sock:
                recv_sock.bind(("127.0.0.1", 0))
                ports.append(recv_sock.getsockname()[1])
                receiver.add_udp_reader(socket=recv_sock)
        return ports

    def prepare_senders(self, thread_pool, n, ports, cls=spead2.send.UdpStream):
        assert len(ports) == n
        # spead2 duplicates the socket, so we use a context manager to
        # close the original.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as send_sock:
            sender = cls(
                thread_pool,
                send_sock,
                [("127.0.0.1", port) for port in ports],
                spead2.send.StreamConfig(rate=1e7),
            )
        return sender


class AsyncUdpCustomSocketTransport(AsyncTransport):
    is_lossy = True

    async def prepare_receivers(self, receivers):
        return UdpCustomSocketTransport.prepare_receivers(self, receivers)

    async def prepare_senders(self, thread_pool, n, ports):
        return UdpCustomSocketTransport.prepare_senders(
            self, thread_pool, n, ports, cls=spead2.send.asyncio.UdpStream
        )


class UdpMulticastTransport(SyncTransport):
    is_lossy = True
    requires_ipv4_multicast = True
    MCAST_GROUP = "239.255.88.88"
    INTERFACE_ADDRESS = "127.0.0.1"

    def prepare_receivers(self, receivers):
        ports = []
        for i, receiver in enumerate(receivers):
            port = self.unused_udp_port()
            receiver.add_udp_reader(
                self.MCAST_GROUP, port, interface_address=self.INTERFACE_ADDRESS
            )
            ports.append(port)
        return ports

    def prepare_senders(self, thread_pool, n, ports):
        assert len(ports) == n
        return spead2.send.UdpStream(
            thread_pool,
            [(self.MCAST_GROUP, port) for port in ports],
            spead2.send.StreamConfig(rate=1e7),
            buffer_size=0,
            ttl=1,
            interface_address=self.INTERFACE_ADDRESS,
        )


class Udp6MulticastTransport(Udp6Transport):
    requires_ipv6_multicast = True
    MCAST_GROUP = "ff14::1234"

    @staticmethod
    def get_interface_index():
        if not hasattr(socket, "if_nametoindex"):
            pytest.skip("socket.if_nametoindex does not exist")
        for iface in sorted(netifaces.interfaces()):  # Sort to give repeatable results
            addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET6, [])
            for addr in addrs:
                if addr["addr"] != "::1":
                    return socket.if_nametoindex(iface)
        pytest.skip("could not find suitable interface for test")

    def prepare_receivers(self, receivers):
        interface_index = self.get_interface_index()
        ports = []
        for i, receiver in enumerate(receivers):
            port = self.unused_udp_port()
            receiver.add_udp_reader(self.MCAST_GROUP, port, interface_index=interface_index)
            ports.append(port)
        return ports, interface_index

    def prepare_senders(self, thread_pool, n, receiver_state):
        ports, interface_index = receiver_state
        assert len(ports) == n
        return spead2.send.UdpStream(
            thread_pool,
            [(self.MCAST_GROUP, port) for port in ports],
            spead2.send.StreamConfig(rate=1e7),
            buffer_size=0,
            ttl=0,
            interface_index=interface_index,
        )


class UdpIbvTransport(SyncTransport):
    is_lossy = True
    MCAST_GROUP = "239.255.88.88"

    def interface_address(self):
        ifaddr = os.getenv("SPEAD2_TEST_IBV_INTERFACE_ADDRESS")
        if not ifaddr:
            pytest.skip("Envar SPEAD2_TEST_IBV_INTERFACE_ADDRESS not set")
        return ifaddr

    def __init__(self):
        super().__init__()
        self._extra_context = None

    def __enter__(self):
        # mlx5 drivers only enable multicast loopback if there are multiple
        # device contexts. The sender and receiver end up sharing one, so we
        # need to explicitly create another.
        if not hasattr(spead2, "IbvContext"):
            pytest.skip("IBV support not compiled in")
        assert self._extra_context is None, "Transport cannot be entered recursively"
        self._extra_context = spead2.IbvContext(self.interface_address())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._extra_context.reset()
        self._extra_context = None

    def prepare_receivers(self, receivers):
        ports = []
        for i, receiver in enumerate(receivers):
            port = self.unused_udp_port()
            receiver.add_udp_ibv_reader(
                spead2.recv.UdpIbvConfig(
                    endpoints=[(self.MCAST_GROUP, port)],
                    interface_address=self.interface_address(),
                )
            )
            ports.append(port)
        return ports

    def prepare_senders(self, thread_pool, n, ports):
        # The buffer size is deliberately reduced so that we test the
        # wrapping once the buffer has been used up
        assert len(ports) == n
        return spead2.send.UdpIbvStream(
            thread_pool,
            spead2.send.StreamConfig(rate=1e7),
            spead2.send.UdpIbvConfig(
                endpoints=[(self.MCAST_GROUP, port) for port in ports],
                interface_address=self.interface_address(),
                buffer_size=64 * 1024,
            ),
        )


class TcpTransport(SyncNoSubstreamsTransport):
    def prepare_receiver(self, receiver, bind_hostname="127.0.0.1"):
        port = self.unused_tcp_port()
        receiver.add_tcp_reader(port, bind_hostname=bind_hostname)
        return (bind_hostname, port)

    def prepare_sender(self, thread_pool, endpoint):
        return spead2.send.TcpStream(thread_pool, [endpoint])


class AsyncTcpTransport(AsyncNoSubstreamsTransport):
    async def prepare_receiver(self, receiver):
        return TcpTransport.prepare_receiver(self, receiver)

    async def prepare_sender(self, thread_pool, endpoint):
        return await spead2.send.asyncio.TcpStream.connect(thread_pool, [endpoint])


class TcpCustomSocketTransport(SyncNoSubstreamsTransport):
    def prepare_receiver(self, receiver):
        # spead2 duplicates the socket, so ensure we close it.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.listen(1)
            receiver.add_tcp_reader(sock)
            return port

    def prepare_sender(self, thread_pool, port):
        with socket.socket() as sock:
            sock.connect(("127.0.0.1", port))
            sender = spead2.send.TcpStream(thread_pool, sock)
        return sender


class AsyncTcpCustomSocketTransport(AsyncNoSubstreamsTransport):
    async def prepare_receiver(self, receiver):
        return TcpCustomSocketTransport.prepare_receiver(self, receiver)

    async def prepare_sender(self, thread_pool, port):
        with socket.socket() as sock:
            sock.setblocking(False)
            await asyncio.get_event_loop().sock_connect(sock, ("127.0.0.1", port))
            sender = spead2.send.asyncio.TcpStream(thread_pool, sock)
        return sender


class Tcp6Transport(TcpTransport):
    requires_ipv6 = True

    def prepare_receiver(self, receiver):
        return super().prepare_receiver(receiver, bind_hostname="::1")

    def prepare_sender(self, thread_pool, endpoint):
        return spead2.send.TcpStream(thread_pool, [endpoint])


class MemTransport(Transport):
    supports_substreams = False

    @classmethod
    def transmit_item_groups(
        cls, item_groups, *, memcpy, allocator, explicit_start, new_order="=", group_mode=None
    ):
        assert len(item_groups) == 1
        assert group_mode is None
        thread_pool = spead2.ThreadPool(2)
        sender = spead2.send.BytesStream(thread_pool)
        gen = spead2.send.HeapGenerator(item_groups[0])
        sender.send_heap(gen.get_heap())
        sender.send_heap(gen.get_end())
        recv_config = spead2.recv.StreamConfig(memcpy=memcpy, explicit_start=explicit_start)
        if allocator is not None:
            recv_config.memory_allocator = allocator
        receiver = spead2.recv.Stream(thread_pool, recv_config)
        receiver.add_buffer_reader(sender.getvalue())
        received_item_group = spead2.ItemGroup()
        if explicit_start:
            receiver.start()
        for heap in receiver:
            received_item_group.update(heap, new_order)
        return [received_item_group]


class InprocTransport(SyncTransport):
    def __init__(self):
        self._queues = []

    def __exit__(self, exc_type, exc_value, traceback):
        for queue in self._queues:
            queue.stop()
        self._queues = []

    def prepare_receivers(self, receivers):
        self._queues = [spead2.InprocQueue() for _ in receivers]
        for receiver, queue in zip(receivers, self._queues):
            receiver.add_inproc_reader(queue)

    def prepare_senders(self, thread_pool, n, receiver_state):
        assert len(self._queues) == n
        return spead2.send.InprocStream(thread_pool, self._queues)


class AsyncInprocTransport(AsyncTransport):
    def __init__(self):
        self._queues = []

    def __exit__(self, exc_type, exc_value, traceback):
        for queue in self._queues:
            queue.stop()
        self._queues = []

    async def prepare_receivers(self, receivers):
        self._queues = [spead2.InprocQueue() for _ in receivers]
        for receiver, queue in zip(receivers, self._queues):
            receiver.add_inproc_reader(queue)

    async def prepare_senders(self, thread_pool, n, receiver_state):
        assert len(self._queues) == n
        return spead2.send.asyncio.InprocStream(thread_pool, self._queues)


def _test_item_groups(
    transport_class,
    item_groups,
    *,
    memcpy=spead2.MEMCPY_STD,
    allocator=None,
    explicit_start=False,
    new_order="=",
    group_mode=None,
):
    received_item_groups = transport_class.transmit_item_groups(
        item_groups,
        memcpy=memcpy,
        allocator=allocator,
        explicit_start=explicit_start,
        new_order=new_order,
        group_mode=group_mode,
    )
    assert len(received_item_groups) == len(item_groups)
    for received_item_group, item_group in zip(received_item_groups, item_groups):
        assert_item_groups_equal(item_group, received_item_group)
        for item in received_item_group.values():
            if item.dtype is not None:
                assert item.value.dtype == item.value.dtype.newbyteorder(new_order)


def _test_item_group(
    transport_class,
    item_group,
    *,
    memcpy=spead2.MEMCPY_STD,
    allocator=None,
    explicit_start=False,
    new_order="=",
    group_mode=None,
):
    _test_item_groups(
        transport_class,
        [item_group],
        memcpy=memcpy,
        allocator=allocator,
        new_order=new_order,
        group_mode=group_mode,
    )


@pytest.mark.timeout(5)
class TestPassthrough:
    """Tests common to all transports and libraries"""

    @pytest.mark.parametrize("explicit_start", [False, True])
    def test_numpy_simple(self, transport_class, explicit_start):
        """A basic array with numpy encoding"""
        ig = spead2.send.ItemGroup()
        data = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.uint16)
        ig.add_item(
            id=0x2345,
            name="name",
            description="description",
            shape=data.shape,
            dtype=data.dtype,
            value=data,
        )
        _test_item_group(transport_class, ig, explicit_start=explicit_start)

    def test_numpy_byteorder(self, transport_class):
        """A numpy array in non-native byte order"""
        ig = spead2.send.ItemGroup()
        data = np.array([[6, 7, 8], [10, 11, 12000]], dtype=np.dtype(np.uint16).newbyteorder())
        ig.add_item(
            id=0x2345,
            name="name",
            description="description",
            shape=data.shape,
            dtype=data.dtype,
            value=data,
        )
        _test_item_group(transport_class, ig)
        _test_item_group(transport_class, ig, new_order="|")

    def test_numpy_large(self, transport_class):
        """A numpy style array split across several packets. It also
        uses non-temporal copies, a custom allocator, and a memory pool,
        to test that those all work."""
        # macOS doesn't have a big enough socket buffer to reliably transmit
        # the whole thing over UDP.
        if transport_class.is_lossy and sys.platform == "darwin":
            pytest.skip("macOS can't reliably handle large heaps over UDP")
        ig = spead2.send.ItemGroup()
        data = np.random.randn(100, 200)
        ig.add_item(
            id=0x2345,
            name="name",
            description="description",
            shape=data.shape,
            dtype=data.dtype,
            value=data,
        )
        allocator = spead2.MmapAllocator()
        pool = spead2.MemoryPool(1, 4096, 4, 4, allocator)
        _test_item_group(transport_class, ig, memcpy=spead2.MEMCPY_NONTEMPORAL, allocator=pool)

    def test_fallback_struct_whole_bytes(self, transport_class):
        """A structure with non-byte-aligned elements, but which is
        byte-aligned overall."""
        ig = spead2.send.ItemGroup()
        format = [("u", 4), ("f", 64), ("i", 4)]
        data = (12, 1.5, -3)
        ig.add_item(
            id=0x2345, name="name", description="description", shape=(), format=format, value=data
        )
        _test_item_group(transport_class, ig)

    def test_string(self, transport_class):
        """Byte string is converted to array of characters and back."""
        ig = spead2.send.ItemGroup()
        format = [("c", 8)]
        data = "Hello world"
        ig.add_item(
            id=0x2345,
            name="name",
            description="description",
            shape=(None,),
            format=format,
            value=data,
        )
        _test_item_group(transport_class, ig)

    def test_fallback_array_partial_bytes_small(self, transport_class):
        """An array which takes a fractional number of bytes per element
        and is small enough to encode in an immediate.
        """
        ig = spead2.send.ItemGroup()
        format = [("u", 7)]
        data = [127, 12, 123]
        ig.add_item(
            id=0x2345,
            name="name",
            description="description",
            shape=(len(data),),
            format=format,
            value=data,
        )
        _test_item_group(transport_class, ig)

    def test_fallback_types(self, transport_class):
        """An array structure using a mix of types."""
        ig = spead2.send.ItemGroup()
        format = [("b", 1), ("i", 7), ("c", 8), ("f", 32)]
        data = [(True, 17, b"y", 1.0), (False, -23, b"n", -1.0)]
        ig.add_item(
            id=0x2345, name="name", description="description", shape=(2,), format=format, value=data
        )
        _test_item_group(transport_class, ig)

    def test_numpy_fallback_struct(self, transport_class):
        """A structure specified using a format, but which is encodable using numpy."""
        ig = spead2.send.ItemGroup()
        format = [("u", 8), ("f", 32)]
        data = (12, 1.5)
        ig.add_item(
            id=0x2345, name="name", description="description", shape=(), format=format, value=data
        )
        _test_item_group(transport_class, ig)

    def test_fallback_struct_partial_bytes(self, transport_class):
        """A structure which takes a fractional number of bytes per element."""
        ig = spead2.send.ItemGroup()
        format = [("u", 4), ("f", 64)]
        data = (12, 1.5)
        ig.add_item(
            id=0x2345, name="name", description="description", shape=(), format=format, value=data
        )
        _test_item_group(transport_class, ig)

    def test_fallback_scalar(self, transport_class):
        """Send a scalar using fallback format descriptor."""
        ig = spead2.send.ItemGroup()
        format = [("f", 64)]
        data = 1.5
        ig.add_item(
            id=0x2345,
            name="scalar name",
            description="scalar description",
            shape=(),
            format=format,
            value=data,
        )
        _test_item_group(transport_class, ig)

    def test_many_items(self, transport_class):
        """Sends many items.

        The implementation handles few-item heaps differently (for
        performance), so need to test that the many-item code works too.
        """
        ig = spead2.send.ItemGroup()
        for i in range(50):
            name = f"test item {i}"
            ig.add_item(
                id=0x2345 + i,
                name=name,
                description=name,
                shape=(),
                format=[("u", 40)],
                value=0x12345 * i,
            )
        _test_item_group(transport_class, ig)


@pytest.mark.timeout(5)
@pytest.mark.transport(filter=lambda cls: cls.supports_substreams)
class TestPassthroughSubstreams:
    """Tests for transports support multiple substreams."""

    def test_substreams(self, transport_class):
        item_groups = []
        for i in range(4):
            ig = spead2.ItemGroup()
            ig.add_item(
                id=0x2345,
                name="int",
                description="an integer",
                shape=(),
                format=[("i", 32)],
                value=i,
            )
            item_groups.append(ig)
        _test_item_groups(transport_class, item_groups)

    @pytest.mark.parametrize("size", [10, 20000])
    @pytest.mark.parametrize(
        "group_mode", [spead2.send.GroupMode.ROUND_ROBIN, spead2.send.GroupMode.SERIAL]
    )
    def test_group_modes(self, transport_class, size, group_mode):
        # The interleaving and substream features are independent, but the
        # test framework is set up for one item group per substream.
        item_groups = []
        for i in range(4):
            value = np.random.randint(0, 256, size=size).astype(np.uint8)
            ig = spead2.ItemGroup()
            ig.add_item(
                id=0x2345,
                name="arr",
                description="a random array",
                shape=(size,),
                dtype="u8",
                value=value,
            )
            item_groups.append(ig)
        _test_item_groups(transport_class, item_groups, group_mode=group_mode)

    @pytest.mark.parametrize(
        "group_mode", [spead2.send.GroupMode.ROUND_ROBIN, spead2.send.GroupMode.SERIAL]
    )
    def test_group_modes_mixed_sizes(self, transport_class, group_mode):
        sizes = [20000, 2000, 40000, 30000]
        item_groups = []
        for size in sizes:
            value = np.random.randint(0, 256, size=size).astype(np.uint8)
            ig = spead2.ItemGroup()
            ig.add_item(
                id=0x2345,
                name="arr",
                description="a random array",
                shape=(size,),
                dtype="u8",
                value=value,
            )
            item_groups.append(ig)
        _test_item_groups(transport_class, item_groups, group_mode=group_mode)


@pytest.mark.timeout(5)
class TestPassthroughUdp:
    def test_empty_endpoints(self):
        with pytest.raises(ValueError):
            spead2.send.UdpStream(spead2.ThreadPool(), [], spead2.send.StreamConfig(rate=1e7))

    def test_mixed_protocols(self):
        with pytest.raises(ValueError):
            spead2.send.UdpStream(
                spead2.ThreadPool(),
                [("127.0.0.1", 0), ("::1", 0)],
                spead2.send.StreamConfig(rate=1e7),
            )


@pytest.mark.timeout(5)
class TestPassthroughUdpIbv:
    @pytest.mark.parametrize("num_items", [0, 1, 3, 4, 10])
    def test_memory_regions(self, num_items, unused_udp_port):
        with UdpIbvTransport() as transport:
            receiver = spead2.recv.Stream(spead2.ThreadPool(), spead2.recv.StreamConfig())
            receiver.add_udp_ibv_reader(
                spead2.recv.UdpIbvConfig(
                    endpoints=[(transport.MCAST_GROUP, unused_udp_port)],
                    interface_address=transport.interface_address(),
                )
            )

            ig = spead2.send.ItemGroup()
            data = [np.random.randn(50) for i in range(num_items)]
            for i in range(num_items):
                ig.add_item(
                    id=0x2345 + i,
                    name=f"name {i}",
                    description=f"description {i}",
                    shape=data[i].shape,
                    dtype=data[i].dtype,
                    value=data[i],
                )
            sender = spead2.send.UdpIbvStream(
                spead2.ThreadPool(),
                spead2.send.StreamConfig(rate=1e7),
                spead2.send.UdpIbvConfig(
                    endpoints=[(transport.MCAST_GROUP, unused_udp_port)],
                    interface_address=transport.interface_address(),
                    memory_regions=data,
                ),
            )
            sender.send_heap(ig.get_heap())
            sender.send_heap(ig.get_end())

            recv_ig = spead2.ItemGroup()
            for heap in receiver:
                recv_ig.update(heap)
            assert_item_groups_equal(ig, recv_ig)


class TestPassthroughInproc:
    def test_queues(self):
        queues = [spead2.InprocQueue() for i in range(2)]
        stream = spead2.send.InprocStream(spead2.ThreadPool(), queues)
        assert stream.queues == queues


TRANSPORT_CLASSES = [
    pytest.param(UdpTransport, id="udp"),
    pytest.param(AsyncUdpTransport, id="async_udp"),
    pytest.param(Udp6Transport, id="udp6"),
    pytest.param(UdpCustomSocketTransport, id="udp_custom"),
    pytest.param(AsyncUdpCustomSocketTransport, id="async_udp_custom"),
    pytest.param(UdpMulticastTransport, id="udp_multicast"),
    pytest.param(Udp6MulticastTransport, id="udp6_multicast"),
    pytest.param(UdpIbvTransport, id="udp_ibv"),
    pytest.param(TcpTransport, id="tcp"),
    pytest.param(AsyncTcpTransport, id="async_tcp"),
    pytest.param(TcpCustomSocketTransport, id="tcp_custom"),
    pytest.param(AsyncTcpCustomSocketTransport, id="async_tcp_custom"),
    pytest.param(Tcp6Transport, id="tcp6"),
    pytest.param(MemTransport, id="mem"),
    pytest.param(InprocTransport, id="inproc"),
    pytest.param(AsyncInprocTransport, id="async_inproc"),
]


@pytest.hookimpl
def pytest_generate_tests(metafunc):
    """Provide the `transport_class` fixture.

    This is a hook rather than just a fixture because it allows marks to
    filter the set of classes.
    """
    if "transport_class" in metafunc.fixturenames:
        transport_classes = TRANSPORT_CLASSES
        for mark in metafunc.definition.iter_markers("transport"):
            filt = mark.kwargs.get("filter")
            if filt is not None:
                transport_classes = [cls for cls in transport_classes if filt(cls.values[0])]
        metafunc.parametrize("transport_class", transport_classes)
