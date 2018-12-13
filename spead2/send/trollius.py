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

from spead2._spead2.send import UdpStreamAsyncio as _UdpStreamAsyncio
from spead2._spead2.send import TcpStreamAsyncio as _TcpStreamAsyncio
from spead2._spead2.send import InprocStreamAsyncio as _InprocStreamAsyncio


def _wrap_class(name, base_class):
    class Wrapped(base_class):
        def __init__(self, *args, **kwargs):
            self._loop = kwargs.pop('loop', None)
            super(Wrapped, self).__init__(*args, **kwargs)
            if self._loop is None:
                self._loop = trollius.get_event_loop()
            self._active = 0
            self._last_queued_future = None

        def async_send_heap(self, heap, cnt=-1, loop=None):
            """Send a heap asynchronously. Note that this is *not* a coroutine:
            it returns a future. Adding the heap to the queue is done
            synchronously, to ensure proper ordering.

            Parameters
            ----------
            heap : :py:class:`spead2.send.Heap`
                Heap to send
            cnt : int, optional
                Heap cnt to send (defaults to auto-incrementing)
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
            queued = super(Wrapped, self).async_send_heap(heap, callback, cnt)
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

    Wrapped.__name__ = name
    return Wrapped


UdpStream = _wrap_class('UdpStream', _UdpStreamAsyncio)
UdpStream.__doc__ = \
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
        Event loop to use (defaults to ``trollius.get_event_loop()``)
    """

_TcpStreamBase = _wrap_class('TcpStream', _TcpStreamAsyncio)


class TcpStream(_TcpStreamBase):
    """SPEAD over TCP with asynchronous connect and sends.

    Most users will use :py:meth:`connect` to asynchronously create a stream.
    The constructor should only be used if you wish to provide your own socket
    and take care of connecting yourself.

    Parameters
    ----------
    thread_pool : :py:class:`spead2.ThreadPool`
        Thread pool handling the I/O
    socket : :py:class:`socket.socket`
        TCP/IP Socket that is already connected to the remote end
    config : :py:class:`spead2.send.StreamConfig`
        Stream configuration
    """

    @classmethod
    @trollius.coroutine
    def connect(cls, *args, **kwargs):
        """Open a connection.

        The arguments are the same as for the constructor of
        :py:class:`spead2.send.TcpStream`.
        """
        loop = kwargs.get('loop')
        if loop is None:
            loop = trollius.get_event_loop()
        future = trollius.Future(loop=loop)

        def callback(arg):
            if not future.done():
                if isinstance(arg, Exception):
                    loop.call_soon_threadsafe(future.set_exception, arg)
                else:
                    loop.call_soon_threadsafe(future.set_result, arg)

        stream = cls(callback, *args, **kwargs)
        yield From(future)
        raise Return(stream)


InprocStream = _wrap_class('InprocStream', _InprocStreamAsyncio)
InprocStream.__doc__ = \
    """SPEAD over reliable in-process transport.

    .. note::

        Data may still be lost if the maximum number of in-flight heaps (set
        in the stream config) is exceeded. Either set this value to more
        heaps than will ever be sent (which will use unbounded memory) or be
        sure to block on the futures returned before exceeding the capacity.

    Parameters
    ----------
    thread_pool : :py:class:`spead2.ThreadPool`
        Thread pool handling the I/O
    queue : :py:class:`spead2.InprocQueue`
        Queue holding the data in flight
    config : :py:class:`spead2.send.StreamConfig`
        Stream configuration
    """

try:
    from spead2._spead2.send import UdpIbvStreamAsyncio as _UdpIbvStreamAsyncio

    UdpIbvStream = _wrap_class('UdpIbvStream', _UdpIbvStreamAsyncio)
    UdpIbvStream.__doc__ = \
        """Like :class:`UdpStream`, but using the Infiniband Verbs API.

        Parameters
        ----------
        thread_pool : :py:class:`spead2.ThreadPool`
            Thread pool handling the I/O
        multicast_group : str
            IP address (or DNS name) of multicast group to join
        port : int
            Peer port
        config : :py:class:`spead2.send.StreamConfig`
            Stream configuration
        interface_address : str
            IP address of network interface from which to send
        buffer_size : int, optional
            Buffer size
        ttl : int, optional
            Time-To-Live of packets
        comp_vector : int, optional
            Completion channel vector (interrupt)
            for asynchronous operation, or
            a negative value to poll continuously. Polling
            should not be used if there are other users of the
            thread pool. If a non-negative value is provided, it
            is taken modulo the number of available completion
            vectors. This allows a number of readers to be
            assigned sequential completion vectors and have them
            load-balanced, without concern for the number
            available.
        max_poll : int
            Maximum number of times to poll in a row, without
            waiting for an interrupt (if `comp_vector` is
            non-negative) or letting other code run on the
            thread (if `comp_vector` is negative).
        loop : :py:class:`trollius.BaseEventLoop`, optional
            Event loop to use (defaults to ``trollius.get_event_loop()``)
        """

except ImportError:
    pass
