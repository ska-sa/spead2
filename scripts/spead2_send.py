#!/usr/bin/env python

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

"""Generate and send SPEAD packets. This is mainly a benchmark application, but also
demonstrates the API."""

from __future__ import print_function, division
import spead2
import spead2.send
import spead2.send.trollius
import numpy as np
import logging
import sys
import itertools
import argparse
import trollius
import collections
from trollius import From


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('host')
    parser.add_argument('port', type=int)

    group = parser.add_argument_group('Data options')
    group.add_argument('--heap-size', metavar='BYTES', type=int, default=4194304, help='Payload size for heap [%(default)s]')
    group.add_argument('--items', type=int, default=1, help='Number of items per heap [%(default)s]')
    group.add_argument('--dtype', type=str, default='<c8', help='Numpy data type [%(default)s]')
    group.add_argument('--heaps', type=int, help='Number of data heaps to send [infinite]')

    group = parser.add_argument_group('Output options')
    group.add_argument('--log', metavar='LEVEL', default='INFO', help='Log level [%(default)s]')

    group = parser.add_argument_group('Protocol options')
    group.add_argument('--pyspead', action='store_true', help='Be bug-compatible with PySPEAD')
    group.add_argument('--addr-bits', type=int, default=40, help='Heap address bits [%(default)s]')
    group.add_argument('--packet', type=int, default=spead2.send.StreamConfig.DEFAULT_MAX_PACKET_SIZE, help='Maximum packet size to send [%(default)s]')
    group.add_argument('--descriptors', type=int, help='Description issue frequency [only at start]')

    group = parser.add_argument_group('Performance options')
    group.add_argument('--buffer', type=int, default=spead2.send.trollius.UdpStream.DEFAULT_BUFFER_SIZE, help='Socket buffer size  [%(default)s]')
    group.add_argument('--threads', type=int, default=1, help='Number of worker threads [%(default)s]')
    group.add_argument('--burst', metavar='BYTES', type=int, default=spead2.send.StreamConfig.DEFAULT_BURST_SIZE, help='Burst size [%(default)s]')
    group.add_argument('--rate', metavar='Gb/s', type=float, default=0, help='Transmission rate bound [no limit]')
    group.add_argument('--affinity', type=spead2.parse_range_list, help='List of CPUs to pin threads to [no affinity]')
    if hasattr(spead2.send, 'UdpIbvStream'):
        group.add_argument('--ibv', type=str, metavar='ADDRESS', help='Use ibverbs with this interface address [no]')
        group.add_argument('--ibv-vector', type=int, default=0, metavar='N', help='Completion vector, or -1 to use polling [%(default)s]')
        group.add_argument('--ibv-max-poll', type=int, default=spead2.send.UdpIbvStream.DEFAULT_MAX_POLL, help='Maximum number of times to poll in a row [%(default)s]')

    return parser.parse_args()


@trollius.coroutine
def run(item_group, stream, args):
    tasks = collections.deque()
    if args.heaps is None:
        rep = itertools.repeat(False)
    else:
        rep = itertools.chain(itertools.repeat(False, args.heaps), [True])
    for is_end in rep:
        if len(tasks) >= 2:
            yield From(trollius.wait([tasks.popleft()]))
        if is_end:
            task = trollius.async(stream.async_send_heap(item_group.get_end()))
        else:
            for item in item_group.values():
                item.version += 1
            task = trollius.async(stream.async_send_heap(item_group.get_heap()))
        tasks.append(task)
    while len(tasks) > 0:
        yield From(trollius.wait([tasks.popleft()]))


def main():
    args = get_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))

    dtype = np.dtype(args.dtype)
    elements = args.heap_size // (args.items * dtype.itemsize)
    heap_size = elements * args.items * dtype.itemsize
    if heap_size != args.heap_size:
        logging.warn('Heap size is not an exact multiple: using %d instead of %d',
                     heap_args, args.heap_size)
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
        rate=args.rate * 1024**3 / 8)
    if 'ibv' in args and args.ibv is not None:
        stream = spead2.send.trollius.UdpIbvStream(
            thread_pool, args.host, args.port, config, args.ibv,
            args.buffer, args.ibv_vector, args.ibv_max_poll)
    else:
        stream = spead2.send.trollius.UdpStream(
            thread_pool, args.host, args.port, config, args.buffer)

    try:
        trollius.get_event_loop().run_until_complete(run(item_group, stream, args))
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == '__main__':
    main()
