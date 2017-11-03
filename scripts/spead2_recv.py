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

"""Receive SPEAD packets and log the contents.

This is both a tool for debugging SPEAD data flows and a demonstrator for the
spead2 package. It thus has many more command-line options than are strictly
necessary, to allow multiple code-paths to be exercised.
"""

import sys


if sys.version_info >= (3, 4):
    from spead2.tools import recv_asyncio
    recv_asyncio.main()
else:
    from spead2.tools import recv_trollius
    recv_trollius.main()
