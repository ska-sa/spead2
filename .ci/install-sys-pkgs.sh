#!/bin/bash
set -e

if [ "$(uname -s)" = "Linux" ]; then
    SUDO=sudo
    if [ `id -u` -eq 0 ]; then
        SUDO=  # We're already root, and sudo might not exist
    fi
    # We may be given something like CXX=sccache g++-NN, and we then want to
    # install g++-NN but not sccache. This splits on whitespace and we end up
    # with the last component.
    for CXX_PKG in $CXX; do :; done
    $SUDO apt-get update && $SUDO apt-get -y --no-install-recommends install \
        "$CXX_PKG" \
        ninja-build \
        gcc \
        lcov \
        libboost-test-dev \
        libboost-program-options-dev \
        libpcap-dev \
        libcap-dev \
        librdmacm-dev \
        libibverbs-dev \
        libdivide-dev
else
    brew update
    brew install ninja boost@1.90 libdivide
    # The MacOS images have an outdated Rust and the line above breaks it.
    brew upgrade rustup
    # On Apple Silicon, homebrew is installed in /opt/homebrew, but the
    # toolchains are not configured to find things there.
    prefix="$(brew --prefix)"
    mkdir -p $HOME/.local/share/meson/native
    cat > $HOME/.local/share/meson/native/ci.ini <<EOF
[properties]
boost_root = '$prefix'

[built-in options]
cpp_args = ['-I$prefix/include']
cpp_link_args = ['-L$prefix/lib']
EOF
fi
