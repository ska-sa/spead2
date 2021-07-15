#!/bin/bash
set -e
for path in /usr/lib/ccache /usr/local/opt/ccache/libexec; do
    if [ -d "$path" ]; then
        echo "$path" >> $GITHUB_PATH
    fi
done
# Clear the statistics so that we can later get statistics for just this run.
ccache -z
