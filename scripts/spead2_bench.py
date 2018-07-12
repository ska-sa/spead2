#!/usr/bin/env python

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


if sys.version_info >= (3, 4):
    from spead2.tools import bench_asyncio
    bench_asyncio.main()
else:
    from spead2.tools import bench_trollius
    bench_trollius.main()
