#!/bin/bash
set -e -x
if [ "$TEST" = "python2" ]; then
    sudo -H pip install -U pip setuptools wheel
    sudo -H pip install -r requirements.txt
    sudo -H pip install "git+https://github.com/ska-sa/PySPEAD#egg=spead"
fi

if [ "$TEST" = "python3" ]; then
    sudo -H pip3 install -U pip setuptools wheel
    sudo -H pip3 install -r requirements.txt
fi

if [ "$TEST" = "cxx" ] && [ "$NETMAP" = "1" ]; then
    git clone https://github.com/luigirizzo/netmap
fi
