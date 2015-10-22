#!/bin/bash
set -e -x
if [ "$TEST" = "python" ]; then
    pip install -r requirements.txt
    if [[ "$TRAVIS_PYTHON_VERSION" == 2.* ]]; then
        pip install "git+https://github.com/ska-sa/PySPEAD#egg=spead"
    fi
fi

if [ "$TEST" = "cxx" ] && [ "$NETMAP" = "1" ]; then
    git clone https://github.com/luigirizzo/netmap
fi
