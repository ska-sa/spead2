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
    if [[ "$PYTHON" == "python3" ]]; then
        python3 --version
        python3 -m venv venv
    fi
elif [[ "$PYTHON" == pypy* ]]; then
    curl -fSL https://bitbucket.org/pypy/pypy/downloads/${PYTHON}-v${PYPY_VERSION}-linux64.tar.bz2 | tar -jx
    PY="$PWD/$PYTHON-v${PYPY_VERSION}-linux64/bin/pypy3"
    virtualenv -p $PY venv
else
    virtualenv -p `which $PYTHON` venv
fi

source venv/bin/activate
pip install -U pip setuptools wheel
if python --version | grep -q 'Python 3\.6'; then
    pip install -r requirements-3.6.txt
else
    pip install -r requirements.txt
fi
if [ "$COVERAGE" = "yes" ]; then
    pip install cpp-coveralls
fi
