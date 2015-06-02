#!/usr/bin/env python

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
    group.add_argument('--heap-size', metavar='BYTES', type=int, default=4194304, help='Payload size for heap')
    group.add_argument('--items', type=int, default=1, help='Number of items per heap')
    group.add_argument('--dtype', type=str, default='<c8', help='Numpy data type')
    group.add_argument('--heaps', type=int, help='Number of data heaps to send (default: infinite)')

    group = parser.add_argument_group('Output options')
    group.add_argument('--log', help='Log configuration file')

    group = parser.add_argument_group('Protocol options')
    group.add_argument('--pyspead', action='store_true', help='Be bug-compatible with PySPEAD')
    group.add_argument('--addr-bits', type=int, default=40, help='Heap address bits')
    group.add_argument('--packet', type=int, default=1472, help='Maximum packet size to send')
    group.add_argument('--descriptors', type=int, help='Description issue frequency')

    group = parser.add_argument_group('Performance options')
    group.add_argument('--buffer', type=int, default=512 * 1024, help='Socket buffer size')
    group.add_argument('--threads', type=int, default=1, help='Number of worker threads')
    group.add_argument('--burst', type=int, default=65536, help='Burst size')
    group.add_argument('--rate', metavar='Gb/s', type=float, default=0, help='Transmission rate bound')

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
    if args.log is not None:
        logging.basicConfig(args.log)
    else:
        logging.basicConfig(level=logging.INFO)

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
    thread_pool = spead2.ThreadPool(args.threads)
    config = spead2.send.StreamConfig(
        max_packet_size=args.packet,
        burst_size=args.burst,
        rate=args.rate * 1024**3 / 8)
    stream = spead2.send.trollius.UdpStream(
        thread_pool, args.host, args.port, config, args.buffer)

    try:
        trollius.get_event_loop().run_until_complete(run(item_group, stream, args))
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == '__main__':
    main()
