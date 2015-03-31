"""
Integration between spead2.send and trollius
"""
from __future__ import absolute_import
import trollius
from trollius import From, Return
import spead2.send
from spead2._send import UdpStreamAsyncio as _UdpStreamAsyncio


class UdpStream(_UdpStreamAsyncio):
    """SPEAD over UDP with asynchronous sends.

    Parameters
    ----------
    thread_pool : :py:class:`spead2.ThreadPool`
        Thread pool handling the I/O
    hostname : str
        Peer hostname
    port : int
        Peer port
    config : :py:class:`spead2.send.StreamConfig`
        Stream configuration
    buffer_size : int
        Socket buffer size. A warning is logged if this size cannot be set due
        to OS limits.
    loop : :py:class:`trollius.BaseEventLoop`, optional
        Event loop to use (defaults to `trollius.get_event_loop()`)
    """
    def __init__(self, *args, **kwargs):
        if 'loop' in kwargs:
            self._loop = kwargs.pop('loop')
        else:
            self._loop = trollius.get_event_loop()
        super(UdpStream, self).__init__(*args, **kwargs)
        self._loop.add_reader(self.fd, self.process_callbacks)

    @trollius.coroutine
    def async_send_heap(self, heap, loop=None):
        """Send a heap asynchronously.

        Parameters
        ----------
        heap : :py:class:`spead2.send.Heap`
            Heap to send
        loop : :py:class:`trollius.BaseEventLoop`, optional
            Event loop to use, overriding the constructor.
        """

        if loop is None:
            loop = self._loop
        future = trollius.Future(loop=self._loop)

        def callback():
            future.set_result(None)
        super(UdpStream, self).async_send_heap(heap, callback)
        yield From(future)

    def __del__(self):
        self._loop.remove_reader(self.fd)
