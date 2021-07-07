#!/bin/bash
set -e

cd /tmp/spead2
mkdir -p /output
version="$(sed 's/.*"\(.*\)"/\1/' src/spead2/_version.py)"
for d in /opt/python/cp{36,37,38,39}*; do
    git clean -xdf
    $d/bin/pip install jinja2==3.0.1 pycparser==2.20 build==0.5.1   # For bootstrap and build
    PATH=$d/bin:$PATH ./bootstrap.sh
    echo "[build_ext]" > setup.cfg
    echo "split_debug = /output" >> setup.cfg
    $d/bin/python -m build
    auditwheel repair --plat manylinux2014_x86_64 -w /output dist/spead2-*-`basename $d`-linux_*.whl
done
cd /output
tar -Jcvf spead2-$version-debug.tar.xz _spead2*.debug
