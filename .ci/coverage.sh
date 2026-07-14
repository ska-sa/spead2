#!/bin/bash
set -e -v

# Collecting the coverage from both the Python and C++ tests is a pain because
# the compiler is run from a different directory in each case, which
# cpp-coveralls doesn't deal with well. lcov handles it better, but ends up
# writing two separate records for each file, which again confuses
# cpp-coveralls (it doesn't merge them). However, if we ask lcov to
# merge the single file it will do merging. We still need --ignore-errors since
# lcov 2.0 because it gets mismatches on the end line for functions.
lcov -c -d . --include "$PWD/include/spead2/*" --include "$PWD/src/*" --ignore-errors mismatch -o lcov-temp.info
lcov -a lcov-temp.info -o lcov.info
