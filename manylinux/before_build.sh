#!/bin/bash
# cibuildwheel script run before each build

set -e -u

# setuptools is pinned to an older version to work around
# https://github.com/pypa/setuptools/issues/3130
pip install jinja2==3.0.3 pycparser==2.21 setuptools==59.8.0
./bootstrap.sh
