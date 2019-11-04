# Copyright 2015, 2019 SKA South Africa
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
Integration between spead2.send and asyncio
"""
from __future__ import absolute_import

import asyncio


from spead2._spead2.send import UdpStreamAsyncio as _UdpStreamAsyncio
from spead2._spead2.send import TcpStreamAsyncio as _TcpStreamAsyncio
from spead2._spead2.send import InprocStreamAsyncio as _InprocStreamAsyncio


class _AsyncHelper:
    def __init__(self, *, loop):
        if loop is None:
            self._loop = asyncio.get_event_loop()
        else:
            self._loop = loop
        self._active = 0
        self._last_queued_future = None

    async def async_send_heap(self, stream, heap, cnt, *extra, loop):
        if loop is None:
            loop = self._loop
        future = asyncio.Future(loop=self._loop)

        def callback(exc, bytes_transferred):
            if exc is not None:
                future.set_exception(exc)
            else:
                future.set_result(bytes_transferred)

        queued = stream.async_send_heap_callback(heap, callback, cnt, *extra)
        if self._active == 0:
            self._loop.add_reader(stream.fd, stream.process_callbacks)
        self._active += 1
        if queued:
            self._last_queued_future = future
        try:
            return await future
        finally:
            self._active -= 1
            if self._active == 0:
                self._loop.remove_reader(stream.fd)
                self._last_queued_future = None  # Purely to free the memory

    async def async_flush(self):
        future = self._last_queued_future
        if future is not None:
            await asyncio.wait([future])


def _fix_docs(cls):
    async_send_heap = getattr(cls, 'async_send_heap')
    if async_send_heap.__doc__ is None:
        async_send_heap.__doc__ = \
            """Send a heap asynchronously.

            Parameters
            ----------
            heap : :py:class:`spead2.send.Heap`
                Heap to send
            cnt : int, optional
                Heap cnt to send (defaults to auto-incrementing)
            loop : :py:class:`asyncio.BaseEventLoop`, optional
                Event loop to use, overriding the constructor.
            """
    async_flush = getattr(cls, 'async_flush')
    if async_flush.__doc__ is None:
        async_flush.__doc__ = \
            """Asynchronously wait for all enqueued heaps to be sent.

            Note that this only waits for heaps passed to
            :meth:`async_send_heap` prior to this call, not ones added while
            waiting."""
    return cls


@_fix_docs
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
    loop : :py:class:`asyncio.BaseEventLoop`, optional
        Event loop to use (defaults to ``asyncio.get_event_loop()``)
    """
    def __init__(self, *args, loop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._helper = _AsyncHelper(loop=loop)

    async def async_send_heap(self, heap, cnt=-1, *args, loop=None, **kwargs):
        """Send a heap asynchronously.

        Parameters
        ----------
        heap : :py:class:`spead2.send.Heap`
            Heap to send
        cnt : int, optional
            Heap cnt to send (defaults to auto-incrementing)
        address : str, optional
            Override the destination IP address. Unlike the constructor, a DNS
            name is not accepted.
        port : int, optional
            Override the destination port. Either both the address and port should
            be specified, or neither.
        loop : :py:class:`asyncio.BaseEventLoop`, optional
            Event loop to use, overriding the constructor.
        """
        return await self._helper.async_send_heap(self, heap, cnt, *args, loop=loop, **kwargs)

    async def async_flush(self):
        await self._helper.async_flush()


@_fix_docs
class TcpStream(_TcpStreamAsyncio):
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
    loop : :py:class:`asyncio.BaseEventLoop`, optional
        Event loop to use (defaults to ``asyncio.get_event_loop()``)
    """

    def __init__(self, *args, loop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._helper = _AsyncHelper(loop=loop)

    @classmethod
    async def connect(cls, *args, **kwargs):
        """Open a connection.

        The arguments are the same as for the constructor of
        :py:class:`spead2.send.TcpStream`.
        """
        loop = kwargs.get('loop')
        if loop is None:
            loop = asyncio.get_event_loop()
        future = asyncio.Future(loop=loop)

        def callback(arg):
            if not future.done():
                if isinstance(arg, Exception):
                    loop.call_soon_threadsafe(future.set_exception, arg)
                else:
                    loop.call_soon_threadsafe(future.set_result, arg)

        stream = cls(callback, *args, **kwargs)
        await future
        return stream

    async def async_send_heap(self, heap, cnt=-1, *, loop=None):
        return await self._helper.async_send_heap(self, heap, cnt, loop=loop)

    async def async_flush(self):
        await self._helper.async_flush()


@_fix_docs
class InprocStream(_InprocStreamAsyncio):
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
    loop : :py:class:`asyncio.BaseEventLoop`, optional
        Event loop to use (defaults to ``asyncio.get_event_loop()``)
    """

    def __init__(self, *args, loop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._helper = _AsyncHelper(loop=loop)

    async def async_send_heap(self, heap, cnt=-1, *, loop=None):
        return await self._helper.async_send_heap(self, heap, cnt, loop=loop)

    async def async_flush(self):
        await self._helper.async_flush()


try:
    from spead2._spead2.send import UdpIbvStreamAsyncio as _UdpIbvStreamAsyncio

    @_fix_docs
    class UdpIbvStream(_UdpIbvStreamAsyncio):
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
        loop : :py:class:`asyncio.BaseEventLoop`, optional
            Event loop to use (defaults to ``asyncio.get_event_loop()``)
        """

        def __init__(self, *args, loop=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._helper = _AsyncHelper(loop=loop)

        async def async_send_heap(self, heap, cnt=-1, *, loop=None):
            return await self._helper.async_send_heap(self, heap, cnt, loop=loop)

        async def async_flush(self):
            await self._helper.async_flush()

except ImportError:
    pass
