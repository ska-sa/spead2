#!/bin/bash
set -e -x

BOOTSTRAP_ARGS=""
PYPY_VERSION=5.9.0
if [[ "$TEST" == pypy* ]]; then
    PY="$PWD/$TEST-v${PYPY_VERSION}-linux64/bin/pypy"
elif [[ "$TEST" == python* ]]; then
    PY="$TEST"
else
    BOOTSTRAP_ARGS="--no-python"
fi

PIP_INSTALL="$PY -m pip install"
if [[ "$TRAVIS_OS_NAME" != osx ]]; then
    PIP_INSTALL="$PIP_INSTALL --user"
fi

./bootstrap.sh $BOOTSTRAP_ARGS
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

if [[ "$TEST" == py* ]]; then
    $PIP_INSTALL -v .
    # Avoid running nosetests from installation directory, to avoid picking up
    # things from the local tree that aren't installed.
    (cd / && nosetests -v spead2)
fi
