# Copyright 2015, 2017, 2019-2020 SKA South Africa
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
                    if args.recv_affinity is not None and len(args.recv_affinity) > 0:
                        spead2.ThreadPool.set_affinity(args.recv_affinity[0])
                        thread_pool = spead2.ThreadPool(
                            1, args.recv_affinity[1:] + args.recv_affinity[:1])
                    else:
                        thread_pool = spead2.ThreadPool()
                    thread_pool = spead2.ThreadPool()
                    memory_pool = spead2.MemoryPool(
                        args.heap_size, args.heap_size + 1024, args.mem_max_free, args.mem_initial)
                    config = spead2.recv.StreamConfig(max_heaps=args.heaps,
                                                      memory_allocator=memory_pool)
                    if args.memcpy_nt:
                        config.memcpy = spead2.MEMCPY_NONTEMPORAL
                    stream = spead2.recv.asyncio.Stream(
                        thread_pool, config, spead2.recv.RingStreamConfig(heaps=args.ring_heaps))
                    bind_hostname = '' if args.multicast is None else args.multicast
                    if 'recv_ibv' in args and args.recv_ibv is not None:
                        try:
                            stream.add_udp_ibv_reader(
                                bind_hostname, args.port,
                                args.recv_ibv,
                                args.packet, args.recv_buffer,
                                args.recv_ibv_vector, args.recv_ibv_max_poll)
                        except AttributeError:
                            logging.error('--recv-ibv passed but agent does not support ibv')
                            sys.exit(1)
                    else:
                        stream.add_udp_reader(args.port, args.packet, args.recv_buffer,
                                              bind_hostname)
                    thread_pool = None
                    memory_pool = None
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
        while len(tasks) >= args.heaps:
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


async def measure_connection_once(args, rate, num_heaps, required_heaps):
    def write(s):
        writer.write(s.encode('ascii'))

    reader, writer = await asyncio.open_connection(args.host, args.port)
    write(json.dumps({'cmd': 'start', 'args': vars(args)}) + '\n')
    # Wait for "ready" response
    response = await reader.readline()
    assert response == b'ready\n'
    if args.send_affinity is not None and len(args.send_affinity) > 0:
        spead2.ThreadPool.set_affinity(args.send_affinity[0])
        thread_pool = spead2.ThreadPool(1, args.send_affinity[1:] + args.send_affinity[:1])
    else:
        thread_pool = spead2.ThreadPool()
    thread_pool = spead2.ThreadPool()
    config = spead2.send.StreamConfig(
        max_packet_size=args.packet,
        burst_size=args.burst,
        rate=rate,
        max_heaps=args.heaps,
        burst_rate_ratio=args.burst_rate_ratio,
        allow_hw_rate=args.allow_hw_rate)
    host = args.host
    if args.multicast is not None:
        host = args.multicast
    if 'send_ibv' in args and args.send_ibv is not None:
        stream = spead2.send.asyncio.UdpIbvStream(
            thread_pool, [(host, args.port)], config, args.send_ibv, args.send_buffer,
            1, args.send_ibv_vector, args.send_ibv_max_poll)
    else:
        stream = spead2.send.asyncio.UdpStream(
            thread_pool, [(host, args.port)], config, args.send_buffer)
    item_group = spead2.send.ItemGroup(
        flavour=spead2.Flavour(4, 64, args.addr_bits, 0))
    item_group.add_item(id=None, name='Test item',
                        description='A test item with arbitrary value',
                        shape=(args.heap_size,), dtype=np.uint8,
                        value=np.zeros((args.heap_size,), dtype=np.uint8))

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


async def measure_connection(args, rate, num_heaps, required_heaps):
    good = True
    rate_sum = 0.0
    passes = 5
    for i in range(passes):
        status, actual_rate = await measure_connection_once(args, rate, num_heaps,
                                                            required_heaps)
        good = good and status
        rate_sum += actual_rate
    return good, rate_sum / passes


async def run_master(args):
    best_actual = 0.0

    # Send 1GB as fast as possible to find an upper bound - receive rate
    # does not matter. Also do a warmup run first to warm up the receiver.
    num_heaps = int(1e9 / args.heap_size) + 2
    await measure_connection_once(args, 0.0, num_heaps, 0)  # warmup
    good, actual_rate = await measure_connection(args, 0.0, num_heaps, num_heaps - 1)
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
            good, actual_rate = await measure_connection(args, rate, num_heaps, num_heaps - 1)
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
    master.add_argument('--quiet', action='store_true', default=False,
                        help='Print only the final result')
    master.add_argument('--packet', metavar='BYTES', type=int, default=9172,
                        help='Maximum packet size to use for UDP [%(default)s]')
    master.add_argument('--heap-size', metavar='BYTES', type=int, default=4194304,
                        help='Payload size for heap [%(default)s]')
    master.add_argument('--addr-bits', metavar='BITS', type=int, default=40,
                        help='Heap address bits [%(default)s]')
    master.add_argument('--multicast', metavar='ADDRESS', type=str,
                        help='Send via multicast group [unicast]')
    group = master.add_argument_group('sender options')
    group.add_argument('--send-affinity', type=spead2.parse_range_list,
                       help='List of CPUs to pin threads to [no affinity]')
    group.add_argument('--send-buffer', metavar='BYTES', type=int,
                       default=spead2.send.asyncio.UdpStream.DEFAULT_BUFFER_SIZE,
                       help='Socket buffer size [%(default)s]')
    group.add_argument('--burst', metavar='BYTES', type=int,
                       default=spead2.send.StreamConfig.DEFAULT_BURST_SIZE,
                       help='Send burst size [%(default)s]')
    group.add_argument('--burst-rate-ratio', metavar='RATIO', type=float,
                       default=spead2.send.StreamConfig.DEFAULT_BURST_RATE_RATIO,
                       help='Hard rate limit, relative to nominal rate [%(default)s]')
    group.add_argument('--no-hw-rate', dest='allow_hw_rate', action='store_false',
                       help='Do not use hardware rate limiting')
    if hasattr(spead2.send, 'UdpIbvStream'):
        group.add_argument('--send-ibv', type=str, metavar='ADDRESS',
                           help='Use ibverbs with this interface address [no]')
        group.add_argument('--send-ibv-vector', type=int, default=0, metavar='N',
                           help='Completion vector, or -1 to use polling [%(default)s]')
        group.add_argument('--send-ibv-max-poll', type=int,
                           default=spead2.send.UdpIbvStream.DEFAULT_MAX_POLL,
                           help='Maximum number of times to poll in a row [%(default)s]')
    group = master.add_argument_group('receiver options')
    group.add_argument('--recv-affinity', type=spead2.parse_range_list,
                       help='List of CPUs to pin threads to [no affinity]')
    group.add_argument('--recv-buffer', metavar='BYTES', type=int,
                       default=spead2.recv.Stream.DEFAULT_UDP_BUFFER_SIZE,
                       help='Socket buffer size [%(default)s]')
    if hasattr(spead2.recv.Stream, 'add_udp_ibv_reader'):
        group.add_argument('--recv-ibv', type=str, metavar='ADDRESS',
                           help='Use ibverbs with this interface address [no]')
        group.add_argument('--recv-ibv-vector', type=int, default=0, metavar='N',
                           help='Completion vector, or -1 to use polling [%(default)s]')
        group.add_argument('--recv-ibv-max-poll', type=int,
                           default=spead2.recv.Stream.DEFAULT_UDP_IBV_MAX_POLL,
                           help='Maximum number of times to poll in a row [%(default)s]')
    group.add_argument('--heaps', type=int, default=spead2.recv.StreamConfig.DEFAULT_MAX_HEAPS,
                       help='Maximum number of in-flight heaps [%(default)s]')
    group.add_argument('--ring-heaps', type=int, default=spead2.recv.RingStreamConfig.DEFAULT_HEAPS,
                       help='Ring buffer capacity in heaps [%(default)s]')
    group.add_argument('--memcpy-nt', action='store_true',
                       help='Use non-temporal memcpy [no]')
    group.add_argument('--mem-max-free', type=int, default=12,
                       help='Maximum free memory buffers [%(default)s]')
    group.add_argument('--mem-initial', type=int, default=8,
                       help='Initial free memory buffers [%(default)s]')
    master.add_argument('host')
    master.add_argument('port', type=int)
    agent = subparsers.add_parser('agent')
    agent.add_argument('port', type=int)

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    if 'host' in args:
        task = run_master(args)
    else:
        task = run_agent(args)
    task = asyncio.ensure_future(task)
    asyncio.get_event_loop().run_until_complete(task)
    task.result()