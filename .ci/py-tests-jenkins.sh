#!/bin/bash
set -e

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

# -ra summarises the reasons for skipping or failing tests
# We suppress the exit code so that the following test publishing step
# is allowed to run if there are failing tests.
setpriv --inh-caps +net_raw --ambient-caps +net_raw -- \
    pytest -v -ra --junitxml=results.xml --suppress-tests-failed-exit-code
