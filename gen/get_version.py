#!/usr/bin/env python3

# Copyright 2020 National Research Foundation (SARAO)
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

import argparse
import re
import sys
from distutils.version import StrictVersion


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('major', 'minor', 'patch', 'full'))
args = parser.parse_args()

with open('src/spead2/_version.py') as version_file:
    line = version_file.readline().strip()
match = re.fullmatch(r'__version__ = "(.+)"', line)
if not match:
    print('src/spead2/_version.py does not match the expected format', file=sys.stderr)
    sys.exit(1)
version_str = match.group(1)
version = StrictVersion(version_str)

mode = args.mode
if mode == 'major':
    print(version.version[0], end='')
elif mode == 'minor':
    print(version.version[1], end='')
elif mode == 'patch':
    print(version.version[2], end='')
else:
    print(version_str, end='')
