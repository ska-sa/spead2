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

"""
Integration between spead2.send and trollius
"""
from __future__ import absolute_import
import trollius
from trollius import From, Return
import spead2.send
from spead2._send import UdpStreamAsyncio as _UdpStreamAsyncio


class UdpStream(_UdpStreamAsyncio):
    """SPEAD over UDP with asynchronous sends. The other constructors
    defined for :py:class:`spead2.send.UdpStream` are also applicable here.

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
        self._loop = kwargs.pop('loop', None)
        if self._loop is None:
            self._loop = trollius.get_event_loop()
        super(UdpStream, self).__init__(*args, **kwargs)
        self._active = 0
        self._last_queued_future = None

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

        def callback(exc, bytes_transferred):
            if exc is not None:
                future.set_exception(exc)
            else:
                future.set_result(bytes_transferred)
            self._active -= 1
            if self._active == 0:
                self._loop.remove_reader(self.fd)
                self._last_queued_future = None  # Purely to free the memory
        queued = super(UdpStream, self).async_send_heap(heap, callback)
        if self._active == 0:
            self._loop.add_reader(self.fd, self.process_callbacks)
        self._active += 1
        if queued:
            self._last_queued_future = future
        return future

    @trollius.coroutine
    def async_flush(self):
        """Asynchronously wait for all enqueued heaps to be sent. Note that
        this only waits for heaps passed to :meth:`async_send_heap` prior to
        this call, not ones added while waiting."""
        future = self._last_queued_future
        if future is not None:
            yield From(trollius.wait([future]))
