#!/bin/bash
# cibuildwheel script run before each build

set -e -u

pip install jinja2==3.0.1 pycparser==2.20
./bootstrap.sh
