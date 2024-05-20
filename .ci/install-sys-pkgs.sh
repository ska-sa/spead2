#!/bin/bash
set -e

# Create this just for consistency with the MacOS native file
mkdir -p $HOME/.local/share/meson/native
touch $HOME/.local/share/meson/native/ci.ini

if [ "$(uname -s)" = "Linux" ]; then
    SUDO=sudo
    if [ `id -u` -eq 0 ]; then
        SUDO=  # We're already root, and sudo might not exist
    fi
    $SUDO apt-get -y install \
        ninja-build \
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
    brew update
    brew install ninja boost@1.85 libdivide
    # On Apple Silicon, homebrew is installed in /opt/homebrew, but the
    # toolchains are not configured to find things there.
    prefix="$(brew --prefix)"
    cat > $HOME/.local/share/meson/native/ci.ini <<EOF
[properties]
boost_root = '$prefix'

[built-in options]
cpp_args = ['-I$prefix/include']
cpp_link_args = ['-L$prefix/lib']
EOF
fi
