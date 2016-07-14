#!/bin/bash
set -e -x

if [[ "$TEST" == "python3" ]]; then
    PIP=pip3
else
    PIP=pip
fi
if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then
    PIP="sudo -H env CC=$CC $PIP"
fi

autoreconf --install
if [ "$TEST" = "cxx" ]; then
    if [ "$NETMAP" = "yes" ]; then
        export CPATH="$PWD/netmap/sys"
    fi
    ./configure \
        --with-netmap="$NETMAP" \
        --with-recvmmsg="$RECVMMSG" \
        --with-eventfd="$EVENTFD" \
        --with-ibv="$IBV" \
        --disable-optimized
    make -j4
    make -j4 check
fi

if [[ "$TEST" == "python2" || "$TEST" == "python3" ]]; then
    # The -e is necessary to make nosetests work, because otherwise it tries to find the
    # .so in the current directory instead of the install directory.
    $PIP install -v -e .
    nosetests -v
fi
