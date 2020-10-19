# Copyright 2015, 2017, 2019-2020 National Research Foundation (SARAO)
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

"""Generate and send SPEAD packets.

This is mainly a benchmark application, but also demonstrates the API.
"""

import logging
import sys
import time
import itertools
import argparse
import collections
import asyncio

import numpy as np

import spead2
import spead2.send
import spead2.send.asyncio
from . import cmdline


def parse_endpoint(endpoint):
    if ':' not in endpoint:
        raise ValueError('destination must have the form <host>:<port>')
    return cmdline.parse_endpoint(endpoint)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=parse_endpoint, nargs='+', metavar='HOST:PORT')

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

    group = parser.add_argument_group('Protocol and performance options')
    protocol = cmdline.ProtocolOptions()
    sender = cmdline.SenderOptions(protocol)
    protocol.add_arguments(group)
    sender.add_arguments(group)
    group.add_argument('--descriptors', type=int,
                       help='Description issue frequency [only at start]')

    args = parser.parse_args()
    protocol.notify(parser, args)
    sender.notify(parser, args)
    if protocol.tcp and len(args.destination) > 1:
        parser.error('only one destination is supported with TCP')
    return args, sender


async def run(item_group, stream, args):
    n_bytes = 0
    n_errors = 0
    last_error = None
    start_time = time.time()
    tasks = collections.deque()
    if args.heaps is None:
        rep = itertools.repeat(False)
    else:
        rep = itertools.chain(
            itertools.repeat(False, args.heaps),
            itertools.repeat(True, len(args.destination))
        )
    n_substreams = stream.num_substreams
    for i, is_end in enumerate(rep):
        if len(tasks) >= args.max_heaps:
            try:
                n_bytes += await tasks.popleft()
            except Exception as error:
                n_errors += 1
                last_error = error
        if is_end:
            heap = item_group.get_end()
        else:
            heap = item_group.get_heap(
                descriptors='all' if i < n_substreams else 'stale',
                data='all')
        task = asyncio.ensure_future(stream.async_send_heap(heap, substream_index=i % n_substreams))
        tasks.append(task)
    while len(tasks) > 0:
        try:
            n_bytes += await tasks.popleft()
        except Exception as error:
            n_errors += 1
            last_error = error
    elapsed = time.time() - start_time
    if last_error is not None:
        logging.warn('%d errors, last one: %s', n_errors, last_error)
    print('Sent {} bytes in {:.6f}s, {:.6f} Gb/s'.format(
        n_bytes, elapsed, n_bytes * 8 / elapsed / 1e9))


async def async_main():
    args, sender = get_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))

    dtype = np.dtype(args.dtype)
    elements = args.heap_size // (args.items * dtype.itemsize)
    heap_size = elements * args.items * dtype.itemsize
    if heap_size != args.heap_size:
        logging.warn('Heap size is not an exact multiple: using %d instead of %d',
                     heap_size, args.heap_size)
    item_group = spead2.send.ItemGroup(
        descriptor_frequency=args.descriptors,
        flavour=sender.make_flavour())
    for i in range(args.items):
        item_group.add_item(id=None, name=f'Test item {i}',
                            description='A test item with arbitrary value',
                            shape=(elements,), dtype=dtype,
                            value=np.zeros((elements,), dtype=dtype))
    thread_pool = sender.make_thread_pool()
    memory_regions = [item.value for item in item_group.values()]
    stream = await sender.make_stream(thread_pool, args.destination, memory_regions)

    await run(item_group, stream, args)


def main():
    try:
        asyncio.get_event_loop().run_until_complete(async_main())
    except KeyboardInterrupt:
        sys.exit(1)
