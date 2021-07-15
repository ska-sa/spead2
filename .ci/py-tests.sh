#!/bin/sh
set -e

# -rs summarises the reasons for skipping tests
pytest -v -rs
for test in test_logging_shutdown test_running_thread_pool test_running_stream; do
    echo "Running shutdown test $test"
    python -c "import tests.shutdown; tests.shutdown.$test()"
done
