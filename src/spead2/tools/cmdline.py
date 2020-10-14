# Copyright 2020 SKA South Africa
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

"""
Command-line processing utilities for the tools in this module.
"""

import spead2.recv
import spead2.send


_HAVE_IBV = hasattr(spead2.recv.Stream, 'add_udp_ibv_reader')


class _Options:
    def __init__(self, name_map=None):
        self._name_map = name_map or {}

    def _add_argument(self, parser, name, *args, **kwargs):
        assert 'dest' not in kwargs
        new_name = self._name_map.get(name, name)
        if new_name is not None:
            flag = '--' + new_name.replace('_', '-')
            parser.add_argument(flag, *args, dest=new_name, default=getattr(self, name), **kwargs)

    def _extract_args(self, namespace):
        for name in self.__dict__:
            if name.startswith('_'):
                continue
            mapped_name = self._name_map.get(name, name)
            if mapped_name is not None:
                setattr(self, name, getattr(namespace, mapped_name))


class ProtocolOptions(_Options):
    """Options that the sender and receiver need to agree on."""

    def __init__(self, name_map=None) -> None:
        super().__init__(name_map)
        self.tcp = False
        self.pyspead = False

    def add_arguments(self, parser):
        self._add_argument(parser, 'tcp', action='store_true',
                           help='Use TCP instead of UDP')
        self._add_argument(parser, 'pyspead', action='store_true',
                           help='Be bug-compatible with PySPEAD')

    def notify(self, parser, namespace):
        self._extract_args(namespace)


class ReceiverOptions(_Options):
    """Options for receivers."""

    def __init__(self, protocol, name_map=None) -> None:
        super().__init__(name_map)
        self._protocol = protocol
        self.memcpy_nt = False
        self.concurrent_heaps = spead2.recv.StreamConfig.DEFAULT_MAX_HEAPS
        self.ring_heaps = spead2.recv.RingStreamConfig.DEFAULT_HEAPS
        self.mem_pool = False
        self.mem_lower = 16384
        self.mem_upper = 32 * 1024**2
        self.mem_max_free = 12
        self.mem_initial = 8
        self.buffer = None
        self.packet = None
        self.bind = None
        if _HAVE_IBV:
            self.ibv = False
            self.ibv_vector = 0
            self.ibv_max_poll = spead2.recv.Stream.DEFAULT_UDP_IBV_MAX_POLL

    def add_arguments(self, parser):
        self._add_argument(parser, 'memcpy_nt', action='store_true',
                           help='Use non-temporal memcpy')
        self._add_argument(parser, 'concurrent_heaps', type=int,
                           help='Maximum number of in-flight heaps [%(default)s]')
        self._add_argument(parser, 'ring_heaps', type=int,
                           help='Ring buffer capacity in heaps [%(default)s]')
        self._add_argument(parser, 'mem_pool', action='store_true', help='Use a memory pool')
        self._add_argument(parser, 'mem_lower', type=int,
                           help='Minimum allocation which will use the memory pool [%(default)s]')
        self._add_argument(parser, 'mem_upper', type=int,
                           help='Maximum allocation which will use the memory pool [%(default)s]')
        self._add_argument(parser, 'mem_max_free', type=int,
                           help='Maximum free memory buffers [%(default)s]')
        self._add_argument(parser, 'mem_initial', type=int,
                           help='Initial free memory buffers [%(default)s]')
        self._add_argument(parser, 'buffer', type=int, help='Socket buffer size')
        self._add_argument(parser, 'packet', type=int, help='Maximum packet size to accept')
        self._add_argument(parser, 'bind', type=str, help='Interface address')
        if _HAVE_IBV:
            self._add_argument(parser, 'ibv', action='store_true', help='Use ibverbs [no]')
            self._add_argument(parser, 'ibv_vector', type=int, metavar='N',
                               help='Completion vector, or -1 to use polling [%(default)s]')
            self._add_argument(parser, 'ibv_max_poll', type=int,
                               help='Maximum number of times to poll in a row [%(default)s]')

    def notify(self, parser, namespace):
        self._extract_args(namespace)
        if _HAVE_IBV:
            if self.ibv and not self.bind:
                parser.error('--ibv requires --bind')
            if self._protocol.tcp and self.ibv:
                parser.error('--ibv and --tcp are incompatible')

        if self.buffer is None:
            if self._protocol.tcp:
                self.buffer = spead2.recv.asyncio.Stream.DEFAULT_TCP_BUFFER_SIZE
            elif _HAVE_IBV and self.ibv:
                self.buffer = spead2.recv.asyncio.Stream.DEFAULT_UDP_IBV_BUFFER_SIZE
            else:
                self.buffer = spead2.recv.asyncio.Stream.DEFAULT_UDP_BUFFER_SIZE

        if self.packet is None:
            if self._protocol.tcp:
                self.packet = spead2.recv.asyncio.Stream.DEFAULT_TCP_MAX_SIZE
            elif _HAVE_IBV and self.ibv:
                self.packet = spead2.recv.asyncio.Stream.DEFAULT_UDP_IBV_MAX_SIZE
            else:
                self.packet = spead2.recv.asyncio.Stream.DEFAULT_UDP_MAX_SIZE

    def make_stream_config(self):
        config = spead2.recv.StreamConfig()
        config.max_heaps = self.concurrent_heaps
        config.bug_compat = spead2.BUG_COMPAT_PYSPEAD_0_5_2 if self._protocol.pyspead else 0
        if self.mem_pool:
            config.memory_allocator = spead2.MemoryPool(self.mem_lower, self.mem_upper,
                                                        self.mem_max_free, self.mem_initial)
        if self.memcpy_nt:
            config.memcpy = spead2.MEMCPY_NONTEMPORAL
        return config

    def make_ring_stream_config(self):
        return spead2.recv.RingStreamConfig(heaps=self.ring_heaps)

    def add_readers(self, stream, endpoints, *, allow_pcap=False):
        ibv_endpoints = []
        for source in endpoints:
            try:
                if ':' in source:
                    host, port = source.rsplit(':', 1)
                    port = int(port)
                else:
                    host = ''
                    port = int(source)
            except ValueError:
                if not allow_pcap:
                    raise
                try:
                    stream.add_udp_pcap_file_reader(source)
                except AttributeError:
                    raise RuntimeError('spead2 was compiled without pcap support') from None
            else:
                if self._protocol.tcp:
                    stream.add_tcp_reader(port, self.packet, self.buffer, host)
                elif _HAVE_IBV and self.ibv:
                    ibv_endpoints.append((host, port))
                elif self.bind:
                    stream.add_udp_reader(host, port, self.packet, self.buffer, self.bind)
                else:
                    stream.add_udp_reader(port, self.packet, self.buffer, host)
        if ibv_endpoints:
            stream.add_udp_ibv_reader(ibv_endpoints, self.bind, self.packet,
                                      self.buffer, self.ibv_vector, self.ibv_max_poll)


class SenderOptions(_Options):
    """Options for senders."""

    def __init__(self, protocol, name_map=None):
        super().__init__(name_map)
        self._protocol = protocol
        self.addr_bits = spead2.Flavour().heap_address_bits
        self.packet = spead2.send.StreamConfig.DEFAULT_MAX_PACKET_SIZE
        self.buffer = None        # Default depends on protocol
        self.bind = None
        self.burst = spead2.send.StreamConfig.DEFAULT_BURST_SIZE
        self.burst_rate_ratio = spead2.send.StreamConfig.DEFAULT_BURST_RATE_RATIO
        self.max_heaps = spead2.send.StreamConfig.DEFAULT_MAX_HEAPS
        self.allow_hw_rate = False
        self.rate = 0.0
        self.ttl = None
        if _HAVE_IBV:
            self.ibv = False
            self.ibv_vector = 0
            self.ibv_max_poll = spead2.send.UdpIbvStreamConfig.DEFAULT_MAX_POLL

    def add_arguments(self, parser):
        self._add_argument(parser, 'addr_bits', type=int,
                           help='Heap address bits [%(default)s]')
        self._add_argument(parser, 'packet', type=int,
                           help='Maximum packet size to send [%(default)s]')
        self._add_argument(parser, 'buffer', type=int, help='Socket buffer size')
        self._add_argument(parser, 'bind', type=str, help='Local address to bind sockets to')
        self._add_argument(parser, 'burst', metavar='BYTES', type=int,
                           help='Burst size [%(default)s]')
        self._add_argument(parser, 'burst_rate_ratio', metavar='RATIO', type=float,
                           help='Hard rate limit, relative to --rate [%(default)s]')
        self._add_argument(parser, 'max_heaps', metavar='HEAPS', type=int,
                           help='Maximum heaps in flight [%(default)s]')
        self._add_argument(parser, 'allow_hw_rate', action='store_true',
                           help='Use hardware rate limiting if available')
        self._add_argument(parser, 'rate', metavar='Gb/s', type=float,
                           help='Transmission rate bound [no limit]')
        self._add_argument(parser, 'ttl', type=int, help='TTL for multicast target')
        if _HAVE_IBV:
            self._add_argument(parser, 'ibv', action='store_true',
                               help='Use ibverbs [no]')
            self._add_argument(parser, 'ibv_vector', type=int, metavar='N',
                               help='Completion vector, or -1 to use polling [%(default)s]')
            self._add_argument(parser, 'ibv_max_poll', type=int,
                               help='Maximum number of times to poll in a row [%(default)s]')

    def notify(self, parser, namespace):
        self._extract_args(namespace)
        if _HAVE_IBV:
            if self.ibv and not self.bind:
                parser.error('--ibv requires --bind')
            if self._protocol.tcp and self.ibv:
                parser.error('--ibv and --tcp are incompatible')
        if self.buffer is None:
            if self._protocol.tcp:
                self.buffer = spead2.send.asyncio.TcpStream.DEFAULT_BUFFER_SIZE
            elif _HAVE_IBV and self.ibv:
                self.buffer = spead2.send.UdpIbvStreamConfig.DEFAULT_BUFFER_SIZE
            else:
                self.buffer = spead2.send.asyncio.UdpStream.DEFAULT_BUFFER_SIZE

    def make_flavour(self):
        bug_compat = spead2.BUG_COMPAT_PYSPEAD_0_5_2 if self._protocol.pyspead else 0
        return spead2.Flavour(4, 64, self.addr_bits, bug_compat)

    def make_stream_config(self):
        return spead2.send.StreamConfig(
            max_packet_size=self.packet,
            rate=self.rate * (1e9 / 8),
            burst_size=self.burst,
            burst_rate_ratio=self.burst_rate_ratio,
            max_heaps=self.max_heaps,
            allow_hw_rate=self.allow_hw_rate)

    async def make_stream(self, thread_pool, endpoints, memory_regions):
        config = self.make_stream_config()
        if self._protocol.tcp:
            return await spead2.send.asyncio.TcpStream.connect(
                thread_pool, endpoints, config, self.buffer, self.bind)
        elif _HAVE_IBV and self.ibv:
            return spead2.send.asyncio.UdpIbvStream(
                thread_pool,
                config,
                spead2.send.UdpIbvStreamConfig(
                    endpoints=endpoints,
                    interface_address=self.bind or '',
                    buffer_size=self.buffer,
                    ttl=self.ttl or 1,
                    comp_vector=self.ibv_vector,
                    max_poll=self.ibv_max_poll,
                    memory_regions=memory_regions
                )
            )
        else:
            kwargs = {}
            if self.ttl is not None:
                kwargs['ttl'] = self.ttl
            if self.bind:
                kwargs['interface_address'] = self.bind
            return spead2.send.asyncio.UdpStream(
                thread_pool, endpoints, config, self.buffer, **kwargs)
