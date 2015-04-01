#!/usr/bin/env python

"""Receive SPEAD packets and log the contents.

This is both a tool for debugging SPEAD data flows and a demonstrator for the
spead2 package. It thus has many more command-line options than are strictly
necessary, to allow multiple code-paths to be exercised.
"""

from __future__ import print_function, division
import spead2
import spead2.recv
import spead2.recv.trollius
import sys
import logging
import argparse
import trollius
from trollius import From

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='+', help='Sources (filenames and port numbers')

    group = parser.add_argument_group('Output options')
    parser.add_argument('--log', help='Log configuration file')
    parser.add_argument('--values', action='store_true', help='Show heap values')

    group = parser.add_argument_group('Protocol options')
    group.add_argument('--pyspead', action='store_true', help='Be bug-compatible with PySPEAD')
    group.add_argument('--joint', action='store_true', help='Treat all sources as a single stream')
    group.add_argument('--packet', type=int, default=9200, help='Maximum packet size to accept for UDP')
    group.add_argument('--buffer', type=int, default=8 * 1024**2, help='Socket buffer size')
    group.add_argument('--bind', default='', help='Bind socket to this hostname')

    group = parser.add_argument_group('Performance options')
    parser.add_argument('--threads', type=int, default=1, help='Number of worker threads')
    parser.add_argument('--heaps', type=int, default=4, help='Maximum number of in-flight heaps')
    parser.add_argument('--mem-pool', action='store_true', help='Use a memory pool')
    parser.add_argument('--mem-lower', type=int, default=16384, help='Minimum allocation which will use the memory pool')
    parser.add_argument('--mem-upper', type=int, default=32 * 1024**2, help='Maximum allocation which will use the memory pool')
    parser.add_argument('--mem-max-free', type=int, default=12, help='Maximum free memory buffers')
    parser.add_argument('--mem-initial', type=int, default=8, help='Initial free memory buffers')
    return parser.parse_args()

@trollius.coroutine
def run_stream(stream, name, args):
    item_group = spead2.ItemGroup()
    num_heaps = 0
    while True:
        try:
            heap = yield From(stream.get())
            print("Received heap {} on stream {}".format(heap.cnt, name))
            num_heaps += 1
            try:
                changed = item_group.update(heap)
                for (key, item) in changed.items():
                    if args.values:
                        print(key, '=', item.value)
                    else:
                        print(key)
            except ValueError as e:
                print("Error raised processing heap: {}".format(e))
        except spead2.Stopped:
            print("Shutting down stream {} after {} heaps".format(name, num_heaps))
            break

def main():
    def make_stream(sources):
        bug_compat = spead2.BUG_COMPAT_PYSPEAD_0_5_2 if args.pyspead else 0
        stream = spead2.recv.trollius.Stream(thread_pool, bug_compat, args.heaps)
        if memory_pool is not None:
            stream.set_memory_pool(stream)
        for source in sources:
            try:
                port = int(source)
            except ValueError:
                with open(source, 'rb') as f:
                    text = f.read()
                stream.add_buffer_reader(text)
            else:
                stream.add_udp_reader(port, args.packet, args.buffer, args.bind)
        return stream

    def make_coro(sources):
        stream = make_stream(sources)
        return run_stream(stream, sources[0], args)

    args = get_args()
    if args.log is not None:
        logging.basicConfig(args.log)
    else:
        logging.basicConfig(level=logging.INFO)

    thread_pool = spead2.ThreadPool(args.threads)
    memory_pool = None
    if args.mem_pool:
        memory_pool = spead2.MemoryPool(args.mem_lower, args.mem_upper, args.mem_max_free, args.mem_initial)
    if args.joint:
        coros = [make_coro(args.source)]
    else:
        coros = [make_coro([source]) for source in args.source]
    tasks = [trollius.async(x) for x in coros]
    task = trollius.wait(tasks, return_when=trollius.FIRST_EXCEPTION)
    trollius.get_event_loop().run_until_complete(task)
    # Trigger any recorded exception
    for t in tasks:
        t.result()

if __name__ == '__main__':
    main()
