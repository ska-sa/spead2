#!/bin/bash

# Prepare a manylinux Docker image for building spead2 wheels (for cibuildwheel)

set -e -u

package="$1"

yum install -y \
    wget libpcap libpcap-devel python-devel \
    cmake3 ninja-build pandoc libnl3-devel \
    ccache

# Install boost
wget --progress=dot:mega https://ufpr.dl.sourceforge.net/project/boost/boost/1.74.0/boost_1_74_0.tar.bz2 -O /tmp/boost_1_74_0.tar.bz2
tar -C /tmp -jxf /tmp/boost_1_74_0.tar.bz2
cd /tmp/boost_1_74_0
./bootstrap.sh --prefix=/usr --with-libraries=program_options,system
./b2 cxxflags=-fPIC link=static install

# Install rdma-core
wget --progress=dot:mega https://github.com/linux-rdma/rdma-core/releases/download/v31.0/rdma-core-31.0.tar.gz -O /tmp/rdma-core-31.0.tar.gz
tar -C /tmp -zxf /tmp/rdma-core-31.0.tar.gz
cd /tmp/rdma-core-31.0
mkdir build
cd build
cmake3 -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja-build -v install
# See https://github.com/pypa/manylinux/issues/731
cp /usr/share/aclocal/pkg.m4 /usr/local/share/aclocal/

# Prepare for split debug symbols for the wheels
cat <<EOF > "$package/setup.cfg"
[build_ext]
split_debug = /output
EOF