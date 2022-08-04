#!/bin/sh
set -e

# -j4 rather than -j to avoid getting killed on Github Actions
if ! make -C build -j4 check; then
  cat build/src/test-suite.log
  exit 1
fi
