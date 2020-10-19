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

"""Benchmark tool to estimate the sustainable SPEAD bandwidth between two
machines, for a specific set of configurations.

Since UDP is lossy, this is not a trivial problem. We binary search for the
speed that is just sustainable. To make the test at a specific speed more
reliable, it is repeated several times, opening a new stream each time, and
with a delay to allow processors to return to idle states. A TCP control
stream is used to synchronise the two ends. All configuration is done on
the master end.
"""

import sys
import argparse
import json
import collections
import logging
import traceback
import timeit
import asyncio

import numpy as np

import spead2
import spead2.recv
import spead2.recv.asyncio
import spead2.send
import spead2.send.asyncio
from . import cmdline


class AgentConnection:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    async def run_stream(self, stream):
        num_heaps = 0
        while True:
            try:
                await stream.get()
                num_heaps += 1
            except spead2.Stopped:
                return num_heaps

    def _write(self, s):
        self.writer.write(s.encode('ascii'))

    async def run_control(self):
        try:
            stream_task = None
            while True:
                command = await self.reader.readline()
                command = command.decode('ascii')
                logging.debug("command = %s", command)
                if not command:
                    break
                command = json.loads(command)
                if command['cmd'] == 'start':
                    if stream_task is not None:
                        logging.warning("Start received while already running: %s", command)
                        continue
                    args = argparse.Namespace(**command['args'])
                    protocol = cmdline.ProtocolOptions()
                    receiver = cmdline.ReceiverOptions(protocol)
                    for (key, value) in command['protocol'].items():
                        setattr(protocol, key, value)
                    for (key, value) in command['receiver'].items():
                        setattr(receiver, key, value)
                    stream = spead2.recv.asyncio.Stream(
                        receiver.make_thread_pool(),
                        receiver.make_stream_config(),
                        receiver.make_ring_stream_config()
                    )
                    if getattr(receiver, 'ibv') and not hasattr(stream, 'add_udp_ibv_reader'):
                        logging.error('--recv-ibv passed but agent does not support ibv')
                        sys.exit(1)

                    endpoint = args.multicast or args.endpoint
                    receiver.add_readers(stream, [endpoint])
                    stream_task = asyncio.ensure_future(self.run_stream(stream))
                    self._write('ready\n')
                elif command['cmd'] == 'stop':
                    if stream_task is None:
                        logging.warning("Stop received when already stopped")
                        continue
                    stream.stop()
                    received_heaps = await stream_task
                    self._write(json.dumps({'received_heaps': received_heaps}) + '\n')
                    stream_task = None
                    stream = None
                elif command['cmd'] == 'exit':
                    break
                else:
                    logging.warning("Bad command: %s", command)
                    continue
            logging.debug("Connection closed")
            if stream_task is not None:
                stream.stop()
                await stream_task
        except Exception:
            traceback.print_exc()


async def agent_connection(reader, writer):
    try:
        conn = AgentConnection(reader, writer)
        await conn.run_control()
    except Exception:
        traceback.print_exc()


async def run_agent(args):
    server = await asyncio.start_server(agent_connection, port=args.port)
    await server.wait_closed()


async def send_stream(item_group, stream, num_heaps, args):
    tasks = collections.deque()
    transferred = 0
    for i in range(num_heaps + 1):
        while len(tasks) >= args.max_heaps:
            transferred += await tasks.popleft()
        if i == num_heaps:
            heap = item_group.get_end()
        else:
            heap = item_group.get_heap(data='all')
        task = asyncio.ensure_future(stream.async_send_heap(heap))
        tasks.append(task)
    for task in tasks:
        transferred += await task
    return transferred


async def measure_connection_once(args, protocol, sender, receiver,
                                  rate, num_heaps, required_heaps):
    def write(s):
        writer.write(s.encode('ascii'))

    host, port = cmdline.parse_endpoint(args.endpoint)
    reader, writer = await asyncio.open_connection(host, port)
    write(json.dumps(
        {
            'cmd': 'start',
            'args': vars(args),
            'protocol': {key: value for (key, value) in protocol.__dict__.items()
                         if not key.startswith('_')},
            'receiver': {key: value for (key, value) in receiver.__dict__.items()
                         if not key.startswith('_')}
        }) + '\n')
    # Wait for "ready" response
    response = await reader.readline()
    assert response == b'ready\n'

    item_group = spead2.send.ItemGroup(flavour=sender.make_flavour())
    item_group.add_item(id=None, name='Test item',
                        description='A test item with arbitrary value',
                        shape=(args.heap_size,), dtype=np.uint8,
                        value=np.zeros((args.heap_size,), dtype=np.uint8))

    sender.rate = rate * 8e-9        # Convert to Gb/s
    endpoint = args.endpoint
    if args.multicast is not None:
        endpoint = args.multicast
    memory_regions = [item.value for item in item_group.values()]
    stream = await sender.make_stream(
        sender.make_thread_pool(), [cmdline.parse_endpoint(endpoint)], memory_regions)

    start = timeit.default_timer()
    transferred = await send_stream(item_group, stream, num_heaps, args)
    end = timeit.default_timer()
    elapsed = end - start
    actual_rate = transferred / elapsed
    # Give receiver time to catch up with any queue
    await asyncio.sleep(0.1)
    write(json.dumps({'cmd': 'stop'}) + '\n')
    # Read number of heaps received
    response = await reader.readline()
    response = json.loads(response.decode('ascii'))
    received_heaps = response['received_heaps']
    await asyncio.sleep(0.5)
    await writer.drain()
    writer.close()
    logging.debug("Received %d/%d heaps in %f seconds, rate %.3f Gbps",
                  received_heaps, num_heaps, elapsed, actual_rate * 8e-9)
    return received_heaps >= required_heaps, actual_rate


async def measure_connection(args, protocol, sender, receiver, rate, num_heaps, required_heaps):
    good = True
    rate_sum = 0.0
    passes = 5
    for i in range(passes):
        status, actual_rate = await measure_connection_once(args, protocol, sender, receiver,
                                                            rate, num_heaps, required_heaps)
        good = good and status
        rate_sum += actual_rate
    return good, rate_sum / passes


async def run_master(args, protocol, sender, receiver):
    best_actual = 0.0

    # Send 1GB as fast as possible to find an upper bound - receive rate
    # does not matter. Also do a warmup run first to warm up the receiver.
    num_heaps = int(1e9 / args.heap_size) + 2
    await measure_connection_once(args, protocol, sender, receiver, 0.0, num_heaps, 0)  # warmup
    good, actual_rate = await measure_connection(args, protocol, sender, receiver,
                                                 0.0, num_heaps, num_heaps - 1)
    if good:
        if not args.quiet:
            print("Limited by send spead")
        best_actual = actual_rate
    else:
        print("Send rate: {:.3f} Gbps".format(actual_rate * 8e-9))
        low = 0.0
        high = actual_rate
        while high - low > high * 0.02:
            # Need at least 1GB of data to overwhelm cache effects, and want at least
            # 1 second for warmup effects.
            rate = (low + high) * 0.5
            num_heaps = int(max(1e9, rate) / args.heap_size) + 2
            good, actual_rate = await measure_connection(
                args, protocol, sender, receiver, rate, num_heaps, num_heaps - 1)
            if not args.quiet:
                print("Rate: {:.3f} Gbps ({:.3f} actual): {}".format(
                    rate * 8e-9, actual_rate * 8e-9, "GOOD" if good else "BAD"))
            if good:
                low = rate
                best_actual = actual_rate
            else:
                high = rate
    rate_gbps = best_actual * 8e-9
    if args.quiet:
        print(rate_gbps)
    else:
        print(f"Sustainable rate: {rate_gbps:.3f} Gbps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', metavar='LEVEL', default='INFO', help='Log level [%(default)s]')
    subparsers = parser.add_subparsers(title='subcommands')

    master = subparsers.add_parser('master')
    sender_map = {
        'buffer': 'send_buffer',
        'packet': None,                 # Use receiver's packet size
        'rate': None,                   # Controlled by test
        'bind': 'send_bind',
        'ibv': 'send_ibv',
        'ibv_vector': 'send_ibv_vector',
        'ibv_max_poll': 'send_ibv_max_poll',
        'affinity': 'send-affinity',
        'threads': 'send-threads'
    }
    receiver_map = {
        'buffer': 'recv_buffer',
        'bind': 'recv_bind',
        'ibv': 'recv_ibv',
        'ibv_vector': 'recv_ibv_vector',
        'ibv_max_poll': 'recv_ibv_max_poll',
        'affinity': 'recv-affinity',
        'threads': 'recv-threads',
        'mem_pool': None,
        'mem_lower': None,
        'mem_upper': None
    }
    protocol = cmdline.ProtocolOptions(name_map={'tcp': None})
    sender = cmdline.SenderOptions(protocol, name_map=sender_map)
    receiver = cmdline.ReceiverOptions(protocol, name_map=receiver_map)
    master.add_argument('--quiet', action='store_true', default=False,
                        help='Print only the final result')
    master.add_argument('--heap-size', metavar='BYTES', type=int, default=4194304,
                        help='Payload size for heap [%(default)s]')
    master.add_argument('--multicast', metavar='ADDRESS', type=str,
                        help='Send via multicast group [unicast]')
    protocol.add_arguments(master)
    sender.add_arguments(master.add_argument_group('sender options'))
    receiver.add_arguments(master.add_argument_group('receiver options'))
    master.add_argument('endpoint', metavar='HOST:PORT')
    agent = subparsers.add_parser('agent')
    agent.add_argument('port', type=int)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    if 'endpoint' in args:
        if args.send_ibv and not args.multicast:
            parser.error('--send-ibv requires --multicast')
        receiver.mem_pool = True
        receiver.mem_lower = args.heap_size
        receiver.mem_upper = args.heap_size + 1024  # more than enough for overheads
        protocol.notify(parser, args)
        receiver.notify(parser, args)
        sender.packet = receiver.packet
        sender.notify(parser, args)
        task = run_master(args, protocol, sender, receiver)
    else:
        task = run_agent(args)
    task = asyncio.ensure_future(task)
    asyncio.get_event_loop().run_until_complete(task)
    task.result()
