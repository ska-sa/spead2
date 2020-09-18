#!/usr/bin/env python3

import argparse
import re
import sys
from distutils.version import StrictVersion


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('major', 'minor', 'patch', 'full'))
args = parser.parse_args()

with open('spead2/_version.py') as version_file:
    line = version_file.readline().strip()
match = re.fullmatch(r'__version__ = "(.+)"', line)
if not match:
    print('spead2/_version.py does not match the expected format', file=sys.stderr)
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
