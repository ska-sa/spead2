# Copyright 2015, 2017, 2019 SKA South Africa
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

"""Generate and send SPEAD packets. This is mainly a benchmark application, but also
demonstrates the API."""

from __future__ import print_function, division
import logging
import sys
import time
import itertools
import argparse
import collections

import numpy as np
import trollius
from trollius import From

import spead2
import spead2.send
import spead2.send.trollius


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('host')
    parser.add_argument('port', type=int)

    group = parser.add_argument_group('Data options')
    group.add_argument('--heap-size', metavar='BYTES', type=int, default=4194304,
                       help='Payload size for heap [%(default)s]')
    group.add_argument('--items', type=int, default=1,
                       help='Number of items per heap [%(default)s]')
    group.add_argument('--dtype', type=str, default='<c8',
                       help='Numpy data type [%(default)s]')
    group.add_argument('--heaps', type=int,
                       help='Number of data heaps to send [infinite]')

    group = parser.add_argument_group('Output options')
    group.add_argument('--log', metavar='LEVEL', default='INFO',
                       help='Log level [%(default)s]')

    group = parser.add_argument_group('Protocol options')
    group.add_argument('--tcp', action='store_true',
                       help='Use TCP instead of UDP')
    group.add_argument('--bind', type=str, default='',
                       help='Local address to bind sockets to')
    group.add_argument('--pyspead', action='store_true',
                       help='Be bug-compatible with PySPEAD')
    group.add_argument('--addr-bits', type=int, default=40,
                       help='Heap address bits [%(default)s]')
    group.add_argument('--packet', type=int,
                       default=spead2.send.StreamConfig.DEFAULT_MAX_PACKET_SIZE,
                       help='Maximum packet size to send [%(default)s]')
    group.add_argument('--descriptors', type=int,
                       help='Description issue frequency [only at start]')

    group = parser.add_argument_group('Performance options')
    group.add_argument('--buffer', type=int,
                       help='Socket buffer size')
    group.add_argument('--threads', type=int, default=1,
                       help='Number of worker threads [%(default)s]')
    group.add_argument('--burst', metavar='BYTES', type=int,
                       default=spead2.send.StreamConfig.DEFAULT_BURST_SIZE,
                       help='Burst size [%(default)s]')
    group.add_argument('--rate', metavar='Gb/s', type=float, default=0,
                       help='Transmission rate bound [no limit]')
    group.add_argument('--burst-rate-ratio', metavar='RATIO', type=float,
                       default=spead2.send.StreamConfig.DEFAULT_BURST_RATE_RATIO,
                       help='Hard rate limit, relative to --rate [%(default)s]')
    group.add_argument('--max-heaps', metavar='HEAPS', type=int,
                       default=spead2.send.StreamConfig.DEFAULT_MAX_HEAPS,
                       help='Maximum heaps in flight')
    group.add_argument('--ttl', type=int, help='TTL for multicast target [1]')
    group.add_argument('--affinity', type=spead2.parse_range_list,
                       help='List of CPUs to pin threads to [no affinity]')
    if hasattr(spead2.send, 'UdpIbvStream'):
        group.add_argument('--ibv', action='store_true',
                           help='Use ibverbs [no]')
        group.add_argument('--ibv-vector', type=int, default=0, metavar='N',
                           help='Completion vector, or -1 to use polling [%(default)s]')
        group.add_argument('--ibv-max-poll', type=int,
                           default=spead2.send.UdpIbvStream.DEFAULT_MAX_POLL,
                           help='Maximum number of times to poll in a row [%(default)s]')

    args = parser.parse_args()
    if args.ibv and not args.bind:
        parser.error('--ibv requires --bind')
    if args.tcp and args.ibv:
        parser.error('--ibv and --tcp are incompatible')
    if args.buffer is None:
        if args.tcp:
            args.buffer = spead2.send.trollius.TcpStream.DEFAULT_BUFFER_SIZE
        else:
            args.buffer = spead2.send.trollius.UdpStream.DEFAULT_BUFFER_SIZE
    return args


@trollius.coroutine
def run(item_group, stream, args):
    n_bytes = 0
    n_errors = 0
    last_error = None
    start_time = time.time()
    tasks = collections.deque()
    if args.heaps is None:
        rep = itertools.repeat(False)
    else:
        rep = itertools.chain(itertools.repeat(False, args.heaps), [True])
    for is_end in rep:
        if len(tasks) >= args.max_heaps:
            try:
                n_bytes += yield From(tasks.popleft())
            except Exception as error:
                n_errors += 1
                last_error = error
        if is_end:
            task = trollius.ensure_future(stream.async_send_heap(item_group.get_end()))
        else:
            for item in item_group.values():
                item.version += 1
            task = trollius.ensure_future(stream.async_send_heap(item_group.get_heap()))
        tasks.append(task)
    while len(tasks) > 0:
        try:
            n_bytes += yield From(tasks.popleft())
        except Exception as error:
            n_errors += 1
            last_error = error
    elapsed = time.time() - start_time
    if last_error is not None:
        logging.warn('%d errors, last one: %s', n_errors, last_error)
    print('Sent {} bytes in {:.6f}s, {:.6f} Gb/s'.format(
        n_bytes, elapsed, n_bytes * 8 / elapsed / 1e9))


@trollius.coroutine
def async_main():
    args = get_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))

    dtype = np.dtype(args.dtype)
    elements = args.heap_size // (args.items * dtype.itemsize)
    heap_size = elements * args.items * dtype.itemsize
    if heap_size != args.heap_size:
        logging.warn('Heap size is not an exact multiple: using %d instead of %d',
                     heap_size, args.heap_size)
    bug_compat = spead2.BUG_COMPAT_PYSPEAD_0_5_2 if args.pyspead else 0
    item_group = spead2.send.ItemGroup(
        descriptor_frequency=args.descriptors,
        flavour=spead2.Flavour(4, 64, args.addr_bits, bug_compat))
    for i in range(args.items):
        item_group.add_item(id=None, name='Test item {}'.format(i),
                            description='A test item with arbitrary value',
                            shape=(elements,), dtype=dtype,
                            value=np.zeros((elements,), dtype=dtype))
    if args.affinity is not None and len(args.affinity) > 0:
        spead2.ThreadPool.set_affinity(args.affinity[0])
        thread_pool = spead2.ThreadPool(args.threads, args.affinity[1:] + args.affinity[:1])
    else:
        thread_pool = spead2.ThreadPool(args.threads)
    config = spead2.send.StreamConfig(
        max_packet_size=args.packet,
        burst_size=args.burst,
        rate=args.rate * 10**9 / 8,
        burst_rate_ratio=args.burst_rate_ratio,
        max_heaps=args.max_heaps)
    if args.tcp:
        stream = yield From(spead2.send.trollius.TcpStream.connect(
            thread_pool, args.host, args.port, config, args.buffer, args.bind))
    elif 'ibv' in args and args.ibv:
        stream = spead2.send.trollius.UdpIbvStream(
            thread_pool, args.host, args.port, config, args.bind,
            args.buffer, args.ttl or 1, args.ibv_vector, args.ibv_max_poll)
    else:
        kwargs = {}
        if args.ttl is not None:
            kwargs['ttl'] = args.ttl
        if args.bind:
            kwargs['interface_address'] = args.bind
        stream = spead2.send.trollius.UdpStream(
            thread_pool, args.host, args.port, config, args.buffer, **kwargs)

    yield From(run(item_group, stream, args))


def main():
    try:
        trollius.get_event_loop().run_until_complete(async_main())
    except KeyboardInterrupt:
        sys.exit(1)
