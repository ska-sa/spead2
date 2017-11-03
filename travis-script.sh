#!/bin/bash
set -e -x

BOOTSTRAP_ARGS=""
if [[ "$TEST" == "python3" ]]; then
    PY="python3"
elif [[ "$TEST" == "python2" ]]; then
    PY="python2"
else
    BOOTSTRAP_ARGS="--no-python"
fi
PIP_INSTALL="$PY -m pip install"
if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then
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

if [[ "$TEST" == "python2" || "$TEST" == "python3" ]]; then
    $PIP_INSTALL -v .
    # Avoid running nosetests from installation directory, to avoid picking up
    # things from the local tree that aren't installed.
    (cd / && nosetests -v spead2)
fi
