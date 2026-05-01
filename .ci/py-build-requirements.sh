#!/bin/bash
set -e -u

# Do some quick-n-dirty parsing to extract the build requirements from
# pyproject.toml.
exec pip install $(grep '^requires = ' pyproject.toml | tr ',' '\n' | sed -n 's/.*"\(.*\)".*/\1/p')
