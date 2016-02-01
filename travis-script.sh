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
    make -j4 -C src CXX="$CXX" AR=ar NETMAP="$NETMAP" RECVMMSG="$RECVMMSG" EVENTFD="$EVENTFD" VARIANT="$VARIANT"
    make -j4 -C src CXX="$CXX" AR=ar NETMAP="$NETMAP" RECVMMSG="$RECVMMSG" EVENTFD="$EVENTFD" VARIANT="$VARIANT" test
fi

if [ "$TEST" = "python" ]; then
    # The -e is necessary to make nosetests work, because otherwise it tries to find the
    # .so in the current directory instead of the install directory.
    CC="$CC" pip install -e .
    nosetests -v
fi
