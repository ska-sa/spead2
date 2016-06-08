#!/bin/bash
set -e -x

if [ "$TEST" = "cxx" ]; then
    if [ "$NETMAP" = "1" ]; then
        export CPATH="$PWD/netmap/sys"
    fi
    if [ "$CXX" = "clang++" ]; then
        VARIANT=debug     # Travis' clang setup is broken for -flto
    else
        VARIANT=release
    fi
    make -j4 -C src CXX="$CXX" AR=ar NETMAP="$NETMAP" RECVMMSG="$RECVMMSG" EVENTFD="$EVENTFD" IBV="$IBV" VARIANT="$VARIANT"
    make -j4 -C src CXX="$CXX" AR=ar NETMAP="$NETMAP" RECVMMSG="$RECVMMSG" EVENTFD="$EVENTFD" IBV="$IBV" VARIANT="$VARIANT" test
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
