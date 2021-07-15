#!/bin/bash
set -e -u

pip install -U pip setuptools wheel
python_version="$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
if [ "$python_version" == "3.6" ]; then
    pip install -r requirements-3.6.txt
else
    pip install -r requirements.txt
fi
