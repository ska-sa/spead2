#!/bin/bash
set -e -x

PYPY_VERSION=5.9.0
if [[ "$TEST" == pypy* ]]; then
    curl -fSL https://bitbucket.org/pypy/pypy/downloads/${TEST}-v${PYPY_VERSION}-linux64.tar.bz2 | tar -jx
    PY="$PWD/$TEST-v${PYPY_VERSION}-linux64/bin/pypy"
    if [ "$TEST" = "pypy3" ]; then
        PY="${PY}3"     # binary is pypy for pypy2 but pypy3 for pypy3
    fi
    $PY -m ensurepip --user
elif [[ "$TEST" == python* ]]; then
    PY="$TEST"
fi

PIP_INSTALL="$PY -m pip install"
if [[ "$TRAVIS_OS_NAME" != osx ]]; then
    PIP_INSTALL="$PIP_INSTALL --user"
fi

if [[ "$TEST" == py* ]]; then
    $PIP_INSTALL -U pip setuptools wheel
    $PIP_INSTALL -r requirements.txt
    if [[ "$TEST" == "python2" ]]; then
        $PIP_INSTALL "git+https://github.com/ska-sa/PySPEAD#egg=spead"
    fi
fi

if [ "$NETMAP" = "yes" ]; then
    git clone https://github.com/luigirizzo/netmap
fi
