#!/bin/bash
set -e -x

if [[ "$TEST" == "python3" ]]; then
    PIP=pip3
else
    PIP=pip
fi
if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then
    PIP="sudo -H $PIP"
fi

if [[ "$TEST" == "python2" || "$TEST" == "python3" ]]; then
    $PIP install -U pip setuptools wheel
    $PIP install -r requirements.txt
    if [[ "$TEST" == "python2" ]]; then
        $PIP install "git+https://github.com/ska-sa/PySPEAD#egg=spead"
    fi
fi

if [ "$NETMAP" = "yes" ]; then
    git clone https://github.com/luigirizzo/netmap
fi
