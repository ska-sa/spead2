#!/bin/bash
# cibuildwheel script run before each build

set -e -u

pip install -c requirements.txt jinja2 pycparser packaging
./bootstrap.sh
