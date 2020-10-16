# Copyright 2015, 2018-2020 National Research Foundation (SARAO)
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
Integration between spead2.recv and asyncio
"""
import collections
import functools
import sys
import asyncio

import spead2.recv


# Decorator from official Python documentation to have compatibility with
# both 3.5.0/1 (which expected __aiter__ to return an awaitable) and 3.5.2+
# (which expects it to return an async iterator).
if sys.version_info < (3, 5, 2):
    def _aiter_compat(func):
        @functools.wraps(func)
        async def wrapper(self):
            return func(self)
        return wrapper
else:
    def _aiter_compat(func):
        return func


class Stream(spead2.recv.Stream):
    """Stream where `get` is a coroutine that yields the next heap.

    Internally, it maintains a queue of waiters, each represented by a future.
    When a heap becomes available, it is passed to the first waiter. We use
    a callback on a file descriptor being readable, which happens when there
    might be data available. The callback is enabled when we have at least one
    waiter, otherwise disabled.

    The futures store a singleton list containing the heap rather than the heap
    itself. This allows the reference to the heap to be explicitly cleared so
    that the heap can be garbage collected sooner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._waiters = collections.deque()
        self._listening = False

    def _start_listening(self):
        if not self._listening:
            asyncio.get_event_loop().add_reader(self.fd, self._ready_callback)
            self._listening = True

    def _stop_listening(self):
        if self._listening:
            asyncio.get_event_loop().remove_reader(self.fd)
            self._listening = False

    def _clear_done_waiters(self):
        """Remove waiters that are done (should only happen if they are cancelled)"""
        while self._waiters and self._waiters[0].done():
            self._waiters.popleft()
        if not self._waiters:
            self._stop_listening()

    def _ready_callback(self):
        self._clear_done_waiters()
        if self._waiters:
            try:
                heap = self.get_nowait()
            except spead2.Empty:
                # Shouldn't happen, but poll may have been woken spuriously
                pass
            except spead2.Stopped as e:
                for waiter in self._waiters:
                    waiter.set_exception(e)
                self._waiters = []
                self._stop_listening()
            else:
                waiter = self._waiters.popleft()
                waiter.set_result([heap])
                if not self._waiters:
                    self._stop_listening()
        # Break cyclic references if spead2.Stopped is raised
        self = None
        waiter = None

    async def get(self):
        """Coroutine that waits for a heap to become available and returns it."""
        self._clear_done_waiters()
        if not self._waiters:
            # If something is available directly, we can avoid going back to
            # the scheduler
            try:
                heap = self.get_nowait()
            except spead2.Empty:
                pass
            else:
                # Give the event loop a chance to run. This ensures that a
                # heap-processing loop cannot live-lock the event loop.
                await asyncio.sleep(0)
                return heap

        waiter = asyncio.Future()
        self._waiters.append(waiter)
        self._start_listening()
        try:
            return (await waiter).pop()
        finally:
            # Prevent cyclic references when an exception is thrown
            waiter = None
            self = None

    @_aiter_compat
    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            heap = await self.get()
        except spead2.Stopped:
            raise StopAsyncIteration
        else:
            return heap
