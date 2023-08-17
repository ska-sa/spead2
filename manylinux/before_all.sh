#!/bin/bash

# Prepare a manylinux Docker image for building spead2 wheels (for cibuildwheel)

set -e -u

package="$1"

yum install -y \
    wget libpcap libpcap-devel \
    cmake3 ninja-build libnl3-devel
if [[ "${CC:-}" == ccache* ]]; then
    yum install -y ccache
fi

# Workaround for https://github.com/pypa/manylinux/issues/1203
unset SSL_CERT_FILE
# The config file sets CFLAGS/LDFLAGS for the actual build, but these break
# building rdma-core
unset CFLAGS
unset LDFLAGS

# Install boost
wget --progress=dot:mega https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.bz2 -O /tmp/boost_1_81_0.tar.bz2
tar -C /tmp -jxf /tmp/boost_1_81_0.tar.bz2
cd /tmp/boost_1_81_0
./bootstrap.sh --prefix=/usr --with-libraries=program_options,system
./b2 cxxflags=-fPIC link=static install

# Install rdma-core
wget --progress=dot:mega https://github.com/linux-rdma/rdma-core/releases/download/v44.0/rdma-core-44.0.tar.gz -O /tmp/rdma-core-44.0.tar.gz
tar -C /tmp -zxf /tmp/rdma-core-44.0.tar.gz
cd /tmp/rdma-core-44.0
mkdir build
cd build
cmake3 -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja-build -v install
# See https://github.com/pypa/manylinux/issues/731
cp /usr/share/aclocal/pkg.m4 /usr/local/share/aclocal/

# Install libdivide
wget https://raw.githubusercontent.com/ridiculousfish/libdivide/5.0/libdivide.h -O /usr/local/include/libdivide.h

# Prepare for split debug symbols for the wheels
cat <<EOF >> "$package/setup.cfg"
[build_ext]
split_debug = /output
EOF
