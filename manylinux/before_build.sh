#!/bin/bash
# cibuildwheel script run before each build

set -e -u

pip install jinja2==3.0.3 pycparser==2.21
./bootstrap.sh
