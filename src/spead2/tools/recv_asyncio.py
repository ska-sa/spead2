# Copyright 2015, 2017-2020 National Research Foundation (SARAO)
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

import logging
import argparse
import signal
import asyncio

import spead2
import spead2.recv
import spead2.recv.asyncio
from . import cmdline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='+', help='Sources (filename, host:port or port)')

    group = parser.add_argument_group('Output options')
    group.add_argument('--log', metavar='LEVEL', default='INFO', help='Log level [%(default)s]')
    group.add_argument('--values', action='store_true', help='Show heap values')
    group.add_argument('--descriptors', action='store_true', help='Show descriptors')

    group = parser.add_argument_group('Input options')
    group.add_argument('--max-heaps', type=int, help='Stop receiving after this many heaps')
    group.add_argument('--joint', action='store_true', help='Treat all sources as a single stream')

    group = parser.add_argument_group('Protocol and performance options')
    protocol = cmdline.ProtocolOptions()
    receiver = cmdline.ReceiverOptions(protocol)
    protocol.add_arguments(group)
    receiver.add_arguments(group)

    args = parser.parse_args()
    protocol.notify(parser, args)
    receiver.notify(parser, args)
    return args, receiver


async def run_stream(stream, name, args):
    try:
        item_group = spead2.ItemGroup()
        num_heaps = 0
        while True:
            try:
                if num_heaps == args.max_heaps:
                    break
                heap = await stream.get()
                print(f"Received heap {heap.cnt} on stream {name}")
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
                    print(f"Error raised processing heap: {e}")
            except (spead2.Stopped, asyncio.CancelledError):
                print(f"Shutting down stream {name} after {num_heaps} heaps")
                stats = stream.stats
                for key in dir(stats):
                    if not key.startswith('_'):
                        print("{}: {}".format(key, getattr(stats, key)))
                break
    finally:
        stream.stop()


def main():
    def make_stream(sources):
        stream = spead2.recv.asyncio.Stream(thread_pool, config, ring_config)
        receiver.add_readers(stream, sources, allow_pcap=True)
        return stream

    def make_coro(sources):
        stream = make_stream(sources)
        return run_stream(stream, sources[0], args), stream

    def stop_streams():
        for stream in streams:
            stream.stop()

    args, receiver = get_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))

    thread_pool = receiver.make_thread_pool()
    config = receiver.make_stream_config()
    ring_config = receiver.make_ring_stream_config()
    if args.joint:
        coros_and_streams = [make_coro(args.source)]
    else:
        coros_and_streams = [make_coro([source]) for source in args.source]
    coros, streams = zip(*coros_and_streams)
    main_task = asyncio.ensure_future(asyncio.gather(*coros))
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, stop_streams)
    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        pass
    loop.close()
