#!/bin/sh
set -e

if ! make -C build -j check; then
  cat build/src/test-suite.log
  exit 1
fi
