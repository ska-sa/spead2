#!/bin/bash
set -e -u

pip install -U pip
if [ "$(python -c 'import sys; print(sys.version_info >= (3, 13))')" == "True" ]; then
    pip install -r requirements-3.13.txt
else
    pip install -r requirements.txt
fi
