#!/bin/bash
set -e -x

autoreconf --install
if [ "$TEST" = "cxx" ]; then
    if [ "$NETMAP" = "yes" ]; then
        export CPATH="$PWD/netmap/sys"
    fi
    AR=ar ./configure \
        --with-netmap="$NETMAP" \
        --with-recvmmsg="$RECVMMSG" \
        --with-eventfd="$EVENTFD" \
        --with-ibv="$IBV"
    make -j4
    make -j4 check
fi

if [ "$TEST" = "python2" ]; then
    # The -e is necessary to make nosetests work, because otherwise it tries to find the
    # .so in the current directory instead of the install directory.
    CC="$CC" sudo -H pip install -e .
    nosetests -v
fi

if [ "$TEST" = "python3" ]; then
    CC="$CC" sudo -H pip3 install -e .
    nosetests -v
fi
