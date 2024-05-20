#!/bin/bash
set -e

if [ "$1" = "--ibverbs" ]; then
    # The force_local_lb_disable file is specific to mlx5. We also want to
    # ensure we pick a device where it is off, since we depend on the loopback
    # for the test.
    device=$(grep -l 'Force local loopback disable is OFF' /sys/class/net/*/settings/force_local_lb_disable | cut -d/ -f5 | head -n1)
    if [ -z "$device" ]; then
        echo "No ibverbs device with local multicast loopback found" 1>&2
        exit 1
    fi
    # Get IPv4 address
    address=$(ip -j addr list dev "$device" | jq -r '.[0].addr_info[] | select(.family == "inet") | .local')
    if [ -z "$address" ]; then
        echo "No IPv4 address found for device $device" 1>&2
        exit 1
    fi
    export SPEAD2_TEST_IBV_INTERFACE_ADDRESS="$address"
    echo "Testing with device $device address $address"
fi

# -ra summarises the reasons for skipping or failing tests
pytest -v -ra
for test in test_logging_shutdown test_running_thread_pool test_running_stream test_running_chunk_stream_group; do
    echo "Running shutdown test $test"
    python -c "import tests.shutdown; tests.shutdown.$test()"
done
