#!/bin/bash
set -e

cd /tmp/spead2
mkdir -p /output
for d in /opt/python/cp{35,36,37,38}*; do
    git clean -xdf
    PATH=$d/bin:$PATH ./bootstrap.sh
    $d/bin/python ./setup.py bdist_wheel -d .
    auditwheel repair --plat manylinux2010_x86_64 -w /output spead2-*-`basename $d`-linux_*.whl
done
