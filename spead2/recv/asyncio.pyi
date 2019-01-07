# Copyright 2019 SKA South Africa
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

import asyncio
from typing import AsyncIterator, Optional

import spead2
import spead2.recv

class Stream(spead2.recv._Stream):
    def __init__(self, thread_pool: spead2.ThreadPool, bug_compat: int = ...,
                 max_heaps: int = ..., ring_heaps: int = ...,
                 contiguous_only: bool = ...,
                 incomplete_keep_payload_ranges: bool = ...,
                 *, loop: Optional[asyncio.AbstractEventLoop] = None) -> None: ...
    async def get(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> spead2.recv.Heap: ...
    def __aiter__(self) -> AsyncIterator[spead2.recv.Heap]: ...
