#!/bin/sh
set -e

sudo docker build --pull -t ska-sa/spead2/manylinux -f manylinux/Dockerfile .
mkdir -p wheelhouse debug-symbols
sudo docker run --rm -v "$PWD/wheelhouse:/wheelhouse" -v "$PWD/debug-symbols:/debug-symbols" ska-sa/spead2/manylinux sh -c '
    cp -v /output/*.whl /wheelhouse
    cp -v /output/*-debug.tar.xz /debug-symbols
'
sudo chown `id -u`:`id -g` wheelhouse/*.whl debug-symbols/*.tar.xz
