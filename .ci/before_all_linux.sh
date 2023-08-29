#!/bin/bash

# Prepare a manylinux Docker image for building spead2 wheels (for cibuildwheel)

set -e -u

sccache_version=0.5.4
rdma_core_version=47.0
pcap_version=1.10.4
boost_version=1.83.0
boost_version_under=${boost_version//./_}

yum install -y wget cmake3 ninja-build flex bison libnl3-devel
# Workaround for https://github.com/pypa/manylinux/issues/1203
unset SSL_CERT_FILE

cd /tmp

# Install sccache
wget --progress=dot:mega https://github.com/mozilla/sccache/releases/download/v${sccache_version}/sccache-v${sccache_version}-$(arch)-unknown-linux-musl.tar.gz
tar -zxf sccache-v${sccache_version}-$(arch)-unknown-linux-musl.tar.gz
cp sccache-v${sccache_version}-$(arch)-unknown-linux-musl/sccache /usr/bin

# Install boost
wget --progress=dot:mega https://boostorg.jfrog.io/artifactory/main/release/${boost_version}/source/boost_${boost_version_under}.tar.bz2
tar -jxf boost_${boost_version_under}.tar.bz2
# Quick-n-dirty approach (much faster than doing the install, which copies thousands of files)
ln -s /tmp/boost_${boost_version_under}/boost /usr/include/boost

# Install rdma-core
wget --progress=dot:mega https://github.com/linux-rdma/rdma-core/releases/download/v${rdma_core_version}/rdma-core-${rdma_core_version}.tar.gz
tar -zxf /tmp/rdma-core-${rdma_core_version}.tar.gz
cd rdma-core-${rdma_core_version}
mkdir build
cd build
cmake3 -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=sccache -DNO_MAN_PAGES=1 -DCMAKE_INSTALL_PREFIX=/usr ..
ninja-build -v install
cd /tmp

# Install pcap (the version provided by yum is old, and it gets vendored into
# the wheel so it's worth keeping up to date).
wget --progress=dot:mega https://www.tcpdump.org/release/libpcap-${pcap_version}.tar.gz
tar -zxf /tmp/libpcap-${pcap_version}.tar.gz
cd libpcap-${pcap_version}/
# CFLAGS is set to avoid generating debug symbols (-g)
./configure --prefix=/usr --disable-rdma --without-libnl CC="sccache gcc" CFLAGS="-O2"
make -j
make install
strip /usr/lib*/libpcap.so.*
cd /tmp

# Install libdivide
wget https://raw.githubusercontent.com/ridiculousfish/libdivide/5.0/libdivide.h -O /usr/local/include/libdivide.h

sccache --show-stats  # Check that sccache is being effective
