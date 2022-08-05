#!/bin/sh
set -e

# -ra summarises the reasons for skipping or failing tests
pytest -v -ra
for test in test_logging_shutdown test_running_thread_pool test_running_stream; do
    echo "Running shutdown test $test"
    python -c "import tests.shutdown; tests.shutdown.$test()"
done
