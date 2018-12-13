# Copyright 2015, 2017-2018 SKA South Africa
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

"""Receive SPEAD packets and log the contents.

This is both a tool for debugging SPEAD data flows and a demonstrator for the
spead2 package. It thus has many more command-line options than are strictly
necessary, to allow multiple code-paths to be exercised.
"""

from __future__ import print_function, division
import logging
import argparse
import signal

import trollius
from trollius import From

import spead2
import spead2.recv
import spead2.recv.trollius


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='+', help='Sources (filename, host:port or port)')

    group = parser.add_argument_group('Output options')
    group.add_argument('--log', metavar='LEVEL', default='INFO', help='Log level [%(default)s]')
    group.add_argument('--values', action='store_true', help='Show heap values')
    group.add_argument('--descriptors', action='store_true', help='Show descriptors')
    group.add_argument('--max-heaps', type=int, help='Stop receiving after this many heaps')

    group = parser.add_argument_group('Protocol options')
    group.add_argument('--tcp', action='store_true', help='Receive data over TCP instead of UDP')
    group.add_argument('--bind', type=str, default='', help='Interface address for multicast')
    group.add_argument('--pyspead', action='store_true', help='Be bug-compatible with PySPEAD')
    group.add_argument('--joint', action='store_true', help='Treat all sources as a single stream')
    group.add_argument('--packet', type=int, default=spead2.recv.Stream.DEFAULT_UDP_MAX_SIZE,
                       help='Maximum packet size to accept for UDP [%(default)s]')

    group = parser.add_argument_group('Performance options')
    group.add_argument('--buffer', type=int, help='Socket buffer size')
    group.add_argument('--threads', type=int, default=1,
                       help='Number of worker threads [%(default)s]')
    group.add_argument('--heaps', type=int, default=spead2.recv.Stream.DEFAULT_MAX_HEAPS,
                       help='Maximum number of in-flight heaps [%(default)s]')
    group.add_argument('--ring-heaps', type=int, default=spead2.recv.Stream.DEFAULT_RING_HEAPS,
                       help='Ring buffer capacity in heaps [%(default)s]')
    group.add_argument('--mem-pool', action='store_true', help='Use a memory pool')
    group.add_argument('--mem-lower', type=int, default=16384,
                       help='Minimum allocation which will use the memory pool [%(default)s]')
    group.add_argument('--mem-upper', type=int, default=32 * 1024**2,
                       help='Maximum allocation which will use the memory pool [%(default)s]')
    group.add_argument('--mem-max-free', type=int, default=12,
                       help='Maximum free memory buffers [%(default)s]')
    group.add_argument('--mem-initial', type=int, default=8,
                       help='Initial free memory buffers [%(default)s]')
    group.add_argument('--memcpy-nt', action='store_true',
                       help='Use non-temporal memcpy')
    group.add_argument('--affinity', type=spead2.parse_range_list,
                       help='List of CPUs to pin threads to [no affinity]')
    if hasattr(spead2.recv.Stream, 'add_udp_ibv_reader'):
        group.add_argument('--ibv', action='store_true', help='Use ibverbs [no]')
        group.add_argument('--ibv-vector', type=int, default=0, metavar='N',
                           help='Completion vector, or -1 to use polling [%(default)s]')
        group.add_argument('--ibv-max-poll', type=int,
                           default=spead2.recv.Stream.DEFAULT_UDP_IBV_MAX_POLL,
                           help='Maximum number of times to poll in a row [%(default)s]')

    args = parser.parse_args()
    if args.ibv and not args.bind:
        parser.error('--ibv requires --bind')
    if args.tcp and args.ibv:
        parser.error('--ibv and --tcp are incompatible')
    if args.buffer is None:
        if args.tcp:
            args.buffer = spead2.recv.trollius.Stream.DEFAULT_TCP_BUFFER_SIZE
        else:
            args.buffer = spead2.recv.trollius.Stream.DEFAULT_UDP_BUFFER_SIZE
    return args


@trollius.coroutine
def run_stream(stream, name, args):
    try:
        item_group = spead2.ItemGroup()
        num_heaps = 0
        while True:
            try:
                if num_heaps == args.max_heaps:
                    break
                heap = yield From(stream.get())
                print("Received heap {} on stream {}".format(heap.cnt, name))
                num_heaps += 1
                try:
                    if args.descriptors:
                        for raw_descriptor in heap.get_descriptors():
                            descriptor = spead2.Descriptor.from_raw(raw_descriptor, heap.flavour)
                            print('''\
    Descriptor for {0.name} ({0.id:#x})
      description: {0.description}
      format:      {0.format}
      dtype:       {0.dtype}
      shape:       {0.shape}'''.format(descriptor))
                    changed = item_group.update(heap)
                    for (key, item) in changed.items():
                        if args.values:
                            print(key, '=', item.value)
                        else:
                            print(key)
                except ValueError as e:
                    print("Error raised processing heap: {}".format(e))
            except (spead2.Stopped, trollius.CancelledError):
                print("Shutting down stream {} after {} heaps".format(name, num_heaps))
                stats = stream.stats
                for key in dir(stats):
                    if not key.startswith('_'):
                        print("{}: {}".format(key, getattr(stats, key)))
                break
    finally:
        stream.stop()


def main():
    def make_stream(sources):
        bug_compat = spead2.BUG_COMPAT_PYSPEAD_0_5_2 if args.pyspead else 0
        stream = spead2.recv.trollius.Stream(thread_pool, bug_compat, args.heaps, args.ring_heaps)
        if memory_pool is not None:
            stream.set_memory_allocator(memory_pool)
        if args.memcpy_nt:
            stream.set_memcpy(spead2.MEMCPY_NONTEMPORAL)
        ibv_endpoints = []
        for source in sources:
            try:
                if ':' in source:
                    host, port = source.rsplit(':', 1)
                    port = int(port)
                else:
                    host = ''
                    port = int(source)
            except ValueError:
                try:
                    stream.add_udp_pcap_file_reader(source)
                except AttributeError:
                    raise RuntimeError('spead2 was compiled without pcap support')
            else:
                if args.tcp:
                    stream.add_tcp_reader(port, args.packet, args.buffer, host)
                elif 'ibv' in args and args.ibv:
                    if host is None:
                        raise ValueError('a multicast group is required when using --ibv')
                    ibv_endpoints.append((host, port))
                elif args.bind and host:
                    stream.add_udp_reader(host, port, args.packet, args.buffer, args.bind)
                else:
                    stream.add_udp_reader(port, args.packet, args.buffer, host)
        if ibv_endpoints:
            stream.add_udp_ibv_reader(ibv_endpoints, args.bind, args.packet,
                                      args.buffer, args.ibv_vector, args.ibv_max_poll)
        return stream

    def make_coro(sources):
        stream = make_stream(sources)
        return run_stream(stream, sources[0], args), stream

    def stop_streams():
        for stream in streams:
            stream.stop()

    args = get_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))

    if args.affinity is not None and len(args.affinity) > 0:
        spead2.ThreadPool.set_affinity(args.affinity[0])
        thread_pool = spead2.ThreadPool(args.threads, args.affinity[1:] + args.affinity[:1])
    else:
        thread_pool = spead2.ThreadPool(args.threads)
    memory_pool = None
    if args.mem_pool:
        memory_pool = spead2.MemoryPool(args.mem_lower, args.mem_upper,
                                        args.mem_max_free, args.mem_initial)
    if args.joint:
        coros_and_streams = [make_coro(args.source)]
    else:
        coros_and_streams = [make_coro([source]) for source in args.source]
    coros, streams = zip(*coros_and_streams)
    main_task = trollius.ensure_future(trollius.gather(*coros))
    loop = trollius.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, stop_streams)
    try:
        loop.run_until_complete(main_task)
    except trollius.CancelledError:
        pass
    loop.close()
