#!/bin/bash
set -e

if [ "$(uname -s)" = "Linux" ]; then
    sudo apt-get install \
        gcc \
        g++ \
        lcov \
        clang \
        libboost-test-dev \
        libboost-program-options-dev \
        libpcap-dev \
        libcap-dev \
        librdmacm-dev \
        libibverbs-dev \
        libdivide-dev
else
    brew install boost libdivide
fi
