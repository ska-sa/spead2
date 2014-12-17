#!/usr/bin/env python
from setuptools import setup, Extension
import glob
import numpy
import sys

bp_library = 'boost_python-py{0}{1}'.format(sys.version_info.major, sys.version_info.minor)

extensions = [
    Extension('_spead2',
        sources=glob.glob('src/common_*.cpp') + glob.glob('src/recv_*.cpp') + glob.glob('src/py_*.cpp'),
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
    version='0.1dev0',
    description='High-performance SPEAD decoder',
    ext_package='spead2',
    ext_modules=extensions,
    install_requires=['numpy'],
    packages=['spead2', 'spead2.recv'])
