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

from setuptools import setup, Extension
import glob
import numpy
import sys

bp_library = 'boost_python-py{0}{1}'.format(sys.version_info.major, sys.version_info.minor)

extensions = [
    Extension(
        '_spead2',
        sources=(glob.glob('src/common_*.cpp') +
                 glob.glob('src/recv_*.cpp') +
                 glob.glob('src/send_*.cpp') +
                 glob.glob('src/py_*.cpp')),
        depends=glob.glob('src/*.h'),
        language='c++',
        include_dirs=['src', numpy.get_include()],
        extra_compile_args=['-std=c++11'],
        libraries=[bp_library, 'boost_system'])
]

setup(
    author='Bruce Merry',
    author_email='bmerry@ska.ac.za',
    name='spead2',
    version='0.1.dev0',
    description='High-performance SPEAD decoder',
    ext_package='spead2',
    ext_modules=extensions,
    install_requires=['numpy'],
    tests_require=['nose'],
    test_suite='nose.collector',
    packages=['spead2', 'spead2.recv', 'spead2.send'])
