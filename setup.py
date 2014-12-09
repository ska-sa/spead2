#!/usr/bin/env python
from distutils.core import setup, Extension

extensions = [
    Extension('_spead2',
        sources=['spead2/pyspead2.cpp', 'src/in.cpp', 'src/udp_in.cpp', 'src/receiver.cpp', 'src/mem_in.cpp'],
        language='c++',
        include_dirs=['src'],
        extra_compile_args=['-std=c++11'],
        libraries=['boost_python-py27', 'boost_system'])
]

setup(
    name='spead2',
    version='0.1dev',
    description='High-performance SPEAD decoder',
    ext_package='spead2',
    ext_modules=extensions)
