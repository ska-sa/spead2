#!/bin/bash
set -e -v

echo "TEST_PYTHON = ${TEST_PYTHON:=no}"
echo "TEST_CXX = ${TEST_CXX:=no}"
echo "COVERAGE = ${COVERAGE:=no}"
echo "PYTHON = ${PYTHON:=python3}"
echo "TRAVIS_OS_NAME = ${TRAVIS_OS_NAME:=linux}"
echo "PYPY_VERSION = ${PYPY_VERSION:=5.9.0}"
echo "CC = ${CC:=gcc}"
echo "CXX = ${CXX:=g++}"

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    if [[ "$PYTHON" == "python2" ]]; then
        virtualenv venv
    elif [[ "$PYTHON" == "python3" ]]; then
        brew update
        brew upgrade python
        pyvenv venv
    fi
elif [[ "$PYTHON" == pypy* ]]; then
    curl -fSL https://bitbucket.org/pypy/pypy/downloads/${PYTHON}-v${PYPY_VERSION}-linux64.tar.bz2 | tar -jx
    PY="$PWD/$PYTHON-v${PYPY_VERSION}-linux64/bin/pypy"
    if [ "$PYTHON" = "pypy3" ]; then
        PY="${PY}3"     # binary is pypy for pypy2 but pypy3 for pypy3
    fi
    virtualenv -p $PY venv
else
    virtualenv -p `which $PYTHON` venv
fi

source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
if [[ "$PYTHON" == "python2" ]]; then
    pip install "git+https://github.com/ska-sa/PySPEAD#egg=spead"
fi
if [ "$COVERAGE" = "yes" ]; then
    pip install cpp-coveralls
fi

if [ "$NETMAP" = "yes" ]; then
    git clone https://github.com/luigirizzo/netmap
    git -C netmap reset --hard 454ef9c   # A known good version
fi
