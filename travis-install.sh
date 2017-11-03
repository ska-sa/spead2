#!/bin/bash
set -e -x

if [[ "$TEST" == "python3" ]]; then
    PY="python3"
else
    PY="python2"
fi
PIP_INSTALL="$PY -m pip install"
if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then
    PIP_INSTALL="$PIP_INSTALL --user"
fi

if [[ "$TEST" == "python2" || "$TEST" == "python3" ]]; then
    $PIP_INSTALL -U pip setuptools wheel
    $PIP_INSTALL -r requirements.txt
    if [[ "$TEST" == "python2" ]]; then
        $PIP_INSTALL "git+https://github.com/ska-sa/PySPEAD#egg=spead"
    fi
fi

if [ "$NETMAP" = "yes" ]; then
    git clone https://github.com/luigirizzo/netmap
fi
