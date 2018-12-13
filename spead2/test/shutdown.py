"""Tests for shutdown ordering.

These cannot be run through nosetests because they deal with interpreter
shutdown.
"""

import logging

import spead2
import spead2.recv
import spead2.send
import spead2._spead2


def test_logging_shutdown():
    """Spam the log with lots of messages, then terminate.

    The logging thread needs to be gracefully cleaned up.
    """
    # Set a log level that won't actually display the messages.
    logging.basicConfig(level=logging.ERROR)
    for i in range(20000):
        spead2._spead2.log_info('Test message {}'.format(i))


def test_running_thread_pool():
    global tp
    tp = spead2.ThreadPool()


def test_running_stream():
    global stream
    logging.basicConfig(level=logging.ERROR)
    stream = spead2.recv.Stream(spead2.ThreadPool())
    stream.add_udp_reader(7148)
    sender = spead2.send.UdpStream(spead2.ThreadPool(), 'localhost', 7148)
    ig = spead2.send.ItemGroup()
    ig.add_item(id=None, name='test', description='test',
                shape=(), format=[('u', 32)], value=0xdeadbeef)
    heap = ig.get_heap()
    for i in range(5):
        sender.send_heap(heap)
