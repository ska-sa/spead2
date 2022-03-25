#!/bin/bash
set -e

if [ "$(uname -s)" = "Linux" ]; then
    sudo apt-get install \
        ccache \
        gcc \
        g++ \
        lcov \
        clang \
        libboost-system-dev \
        libboost-test-dev \
        libboost-program-options-dev \
        libpcap-dev \
        libcap-dev \
        librdmacm-dev \
        libibverbs-dev \
        libdivide-dev
else
    brew install autoconf automake boost ccache
    wget https://raw.githubusercontent.com/ridiculousfish/libdivide/5.0/libdivide.h -O /usr/local/include/libdivide.h
fi
