#!/bin/sh
set -e

sudo docker build --pull -t ska-sa/spead2/manylinux -f manylinux/Dockerfile .
mkdir -p wheelhouse
sudo docker run --rm -v "$PWD/wheelhouse:/wheelhouse" ska-sa/spead2/manylinux sh -c 'cp -v /output/*.whl /wheelhouse'
sudo chown `id -u`:`id -g` wheelhouse/*.whl
