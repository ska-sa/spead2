#!/bin/bash
set -e -u

pcap_version=1.10.5

# Prepare the MacOS environment for building spead2 wheels (for cibuildwheel)
.ci/install-sys-pkgs.sh

# Install libpcap from source
cd /tmp
curl -fsSLO https://www.tcpdump.org/release/libpcap-${pcap_version}.tar.gz
tar -zxf /tmp/libpcap-${pcap_version}.tar.gz
cd libpcap-${pcap_version}/
mkdir build
cd build
prefix="$(brew --prefix)"
# CFLAGS is set to avoid generating debug symbols (-g)
../configure --prefix="$prefix" --disable-rdma --disable-universal --without-libnl \
    CC="sccache cc" CFLAGS="-O2"
make -j
make install
strip -S -x "$prefix"/lib*/libpcap.*
