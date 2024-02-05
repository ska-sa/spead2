#!/bin/bash
set -e -u

pip install -U pip
if [ "$(python -c 'import sys; print(sys.version_info >= (3, 12))')" == "True" ]; then
    pip install -r requirements-3.12.txt
else
    pip install -r requirements.txt
fi
