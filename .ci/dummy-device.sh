#!/bin/bash
# Create a dummy network device with multicast enabled. This is used on
# Github Actions to provide a known-good device for IPv6 multicast tests.
# IPv6 multicast doesn't work on the loopback device by default, and some
# runners in Github Actions have weird device setups that don't work if
# one just picks the first device with an IPv6 address.

set -e

sudo ip link add ci type dummy
sudo ip link set ci multicast on up
