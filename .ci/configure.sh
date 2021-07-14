#!/bin/bash
# Called with either 'yes' or 'no' to indicate whether optional features should
# be used. Any remaining arguments are passed to configure
set -e -u

mkdir -p build
cd build
extras=$1
shift
../configure \
    --with-recvmmsg=$extras \
    --with-sendmmsg=$extras \
    --with-eventfd=$extras \
    --with-ibv=$extras \
    --with-ibv-hw-rate-limit=$extras \
    --with-mlx5dv=$extras \
    --with-pcap=$extras \
    --with-cap=$extras \
    "$@"
