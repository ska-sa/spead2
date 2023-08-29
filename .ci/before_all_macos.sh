#!/bin/bash
set -e -u

pcap_version=1.10.4

# Prepare the MacOS environment for building spead2 wheels (for cibuildwheel)
brew install boost@1.82 libdivide

# Install pkgconfig from source - once for each architecture in CIBW_ARCHS
cd /tmp
wget --progress=dot:mega https://www.tcpdump.org/release/libpcap-${pcap_version}.tar.gz
tar -zxf /tmp/libpcap-${pcap_version}.tar.gz
cd libpcap-${pcap_version}/
for arch in $CIBW_ARCHS; do
    mkdir build-$arch
    cd build-$arch
    if [ "$arch" == $(arch) ]; then
        host_args=""
    else
        host_args="--host=$arch-apple-darwin"
    fi
    # CFLAGS is set to avoid generating debug symbols (-g)
    ../configure $host_args --prefix=/tmp/cibuildwheel/$arch --disable-rdma --disable-universal --without-libnl \
        CC="sccache cc -arch $arch" CFLAGS="-O2"
    make -j
    make install
    strip -S -x /tmp/cibuildwheel/$arch/lib*/libpcap.*
    cd ..
done

# Set up meson cross file to pick up pcap-config. Meson refuses to use the one
# in $PATH when cross-compiling.
cat > /tmp/cibuildwheel/pcap-cross.ini <<EOF
[binaries]
pcap-config = '/tmp/cibuildwheel/$CIBW_ARCHS/bin/pcap-config'
strip = 'strip'
EOF
