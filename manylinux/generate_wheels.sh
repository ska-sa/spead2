#!/bin/bash
set -e

cd /tmp/spead2
mkdir -p /output
version="$(sed 's/.*"\(.*\)"/\1/' src/spead2/_version.py)"
for d in /opt/python/cp{36,37,38,39}*; do
    git clean -xdf
    $d/bin/pip install jinja2==2.10.1 pycparser==2.19   # For bootstrap
    PATH=$d/bin:$PATH ./bootstrap.sh
    echo "[build_ext]" > setup.cfg
    echo "split_debug = /output" >> setup.cfg
    $d/bin/pip wheel --no-deps -v .
    auditwheel repair --plat manylinux2014_x86_64 -w /output spead2-*-`basename $d`-linux_*.whl
done
cd /output
tar -Jcvf spead2-$version-debug.tar.xz _spead2*.debug
