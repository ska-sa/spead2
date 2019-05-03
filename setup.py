#!/usr/bin/env python

# Copyright 2015, 2017, 2019 SKA South Africa
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
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
import glob
import sys
import os
import os.path
import subprocess


def find_version():
    # Cannot simply import it, since that tries to import spead2 as well, which
    # isn't built yet.
    globals_ = {}
    with open(os.path.join(os.path.dirname(__file__), 'spead2', '_version.py')) as f:
        code = f.read()
    exec(code, globals_)
    return globals_['__version__']


# Restrict installed modules to those appropriate to the Python version
class BuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        # distutils uses old-style classes, so no super
        modules = build_py.find_package_modules(self, package, package_dir)
        if sys.version_info < (3, 4):
            modules = [m for m in modules if not m[1].endswith('asyncio')]
        if sys.version_info < (3, 5):
            modules = [m for m in modules if not m[1].endswith('py35')]
        if sys.version_info >= (3, 7):
            modules = [m for m in modules if not m[1].endswith('trollius')]
        return modules


class BuildExt(build_ext):
    user_options = build_ext.user_options + [
        ('coverage', None,
         "build with GCC --coverage option")
    ]
    boolean_options = build_ext.boolean_options + ['coverage']

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.coverage = None

    def run(self):
        self.mkpath(self.build_temp)
        subprocess.check_call(os.path.abspath('configure'), cwd=self.build_temp)
        # Ugly hack to add libraries conditional on configure result
        have_pcap = False
        with open(os.path.join(self.build_temp, 'include', 'spead2', 'common_features.h')) as f:
            for line in f:
                if line.strip() == '#define SPEAD2_USE_PCAP 1':
                    have_pcap = True
        for extension in self.extensions:
            if have_pcap:
                extension.libraries.extend(['pcap'])
            if self.coverage:
                extension.extra_compile_args.extend(['-g', '--coverage'])
                extension.libraries.extend(['gcov'])
            extension.include_dirs.insert(0, os.path.join(self.build_temp, 'include'))
        # distutils uses old-style classes, so no super
        build_ext.run(self)

    def build_extensions(self):
        # Stop GCC complaining about -Wstrict-prototypes in C++ code
        try:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except ValueError:
            pass
        build_ext.build_extensions(self)


# Can't actually install on readthedocs.org because Boost.Python is missing,
# but we need setup.py to still be successful to make the doc build work.
rtd = os.environ.get('READTHEDOCS') == 'True'

if not rtd:
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'configure')):
        raise SystemExit("configure not found. Either download a release " +
                         "from https://pypi.python.org/pypi/spead2 or run " +
                         "./bootstrap.sh if not using a release.")
    if not os.path.exists(os.path.join(
            os.path.dirname(__file__),
            '3rdparty', 'pybind11', 'include', 'pybind11', 'pybind11.h')):
        raise SystemExit("pybind11 not found. Either download a release " +
                         "from https://pypi.python.org/pypi/spead2 or run " +
                         "git submodule update --init --recursive if not " +
                         "using a release.")

    libraries = ['boost_system']

    extensions = [
        Extension(
            '_spead2',
            sources=(glob.glob('src/common_*.cpp') +
                     glob.glob('src/recv_*.cpp') +
                     glob.glob('src/send_*.cpp') +
                     glob.glob('src/py_*.cpp')),
            depends=glob.glob('include/spead2/*.h'),
            language='c++',
            include_dirs=['include', '3rdparty/pybind11/include'],
            extra_compile_args=['-std=c++11', '-g0', '-fvisibility=hidden'],
            libraries=libraries)
    ]
else:
    extensions = []

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme_file:
    readme = readme_file.read()

setup(
    author='Bruce Merry',
    author_email='bmerry@ska.ac.za',
    name='spead2',
    version=find_version(),
    description='High-performance SPEAD implementation',
    long_description=readme,
    url='https://github.com/ska-sa/spead2',
    license='LGPLv3+',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Framework :: AsyncIO',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        'Topic :: System :: Networking'],
    ext_package='spead2',
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExt, 'build_py': BuildPy},
    install_requires=[
        'numpy>=1.9.2',
        'six',
        'trollius; python_version<"3.4"'
    ],
    tests_require=[
        'netifaces',
        'nose',
        'decorator',
        'trollius; python_version<"3.7"',
        'asynctest; python_version>="3.5"'
    ],
    test_suite='nose.collector',
    packages=find_packages(),
    package_data={'': ['py.typed', '*.pyi']},
    scripts=glob.glob('scripts/*.py')
)
