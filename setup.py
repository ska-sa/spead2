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

from __future__ import print_function
from setuptools import setup, Extension
from distutils.command.build_ext import build_ext
import glob
import sys
import os
import os.path
import ctypes.util
import re
import subprocess


try:
    import numpy
    numpy_include = numpy.get_include()
except ImportError:
    numpy_include = None


def find_version():
    # Cannot simply import it, since that tries to import spead2 as well, which
    # isn't built yet.
    globals_ = {}
    with open(os.path.join(os.path.dirname(__file__), 'spead2', '_version.py')) as f:
        code = f.read()
    exec(code, globals_)
    return globals_['__version__']


class BuildExt(build_ext):
    def run(self):
        self.mkpath(self.build_temp)
        subprocess.check_call(os.path.abspath('configure'), cwd=self.build_temp)
        # Ugly hack to add libraries conditional on configure result
        have_ibv = False
        with open(os.path.join(self.build_temp, 'include', 'spead2', 'common_features.h')) as f:
            for line in f:
                if line.strip() == '#define SPEAD2_USE_IBV 1':
                    have_ibv = True
        for extension in self.extensions:
            if have_ibv:
                extension.libraries.extend(['rdmacm', 'ibverbs'])
            extension.include_dirs.insert(0, os.path.join(self.build_temp, 'include'))
        # distutils uses old-style classes, so no super
        build_ext.run(self)


# Can't actually install on readthedocs.org because Boost.Python is missing,
# but we need setup.py to still be successful to make the doc build work.
rtd = os.environ.get('READTHEDOCS') == 'True'

if not rtd:
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'configure')):
        raise SystemExit("configure not found. Either download a release " +
                         "from https://pypi.python.org/pypi/spead2 or run " +
                         "./bootstrap.sh if not using a release.")

    # Different OSes install the Boost.Python library under different names
    bp_library_names = [
        'boost_python-py{0}{1}'.format(sys.version_info.major, sys.version_info.minor),
        'boost_python{0}'.format(sys.version_info.major),
        'boost_python',
        'boost_python-mt']
    for name in bp_library_names:
        if ctypes.util.find_library(name):
            bp_library = name
            break
    else:
        raise RuntimeError('Cannot find Boost.Python library')

    libraries = [bp_library, 'boost_system']

    extensions = [
        Extension(
            '_spead2',
            sources=(glob.glob('src/common_*.cpp') +
                     glob.glob('src/recv_*.cpp') +
                     glob.glob('src/send_*.cpp') +
                     glob.glob('src/py_*.cpp')),
            depends=glob.glob('src/*.h'),
            language='c++',
            include_dirs=['include', numpy_include],
            extra_compile_args=['-std=c++11', '-g0'],
            libraries=libraries)
    ]
else:
    extensions = []

setup(
    author='Bruce Merry',
    author_email='bmerry@ska.ac.za',
    name='spead2',
    version=find_version(),
    description='High-performance SPEAD implementation',
    url='https://github.com/ska-sa/spead2',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        'Topic :: System :: Networking'],
    ext_package='spead2',
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExt},
    setup_requires=['numpy'],
    install_requires=['numpy>=1.9.2', 'six'],
    tests_require=['netifaces', 'nose', 'decorator', 'trollius'],
    test_suite='nose.collector',
    packages=['spead2', 'spead2.recv', 'spead2.send'],
    scripts=glob.glob('scripts/*.py')
)
