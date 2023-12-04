#!/bin/bash
set -e -u

# Build almost all the combinations, to test for interactions that prevent
# compilation (testing them all would take too much time). The loop order
# tries to minimise the changes in the inner loop to speed things up.
meson setup build --werror
cd build
for python in true false; do
    for ibv in auto disabled; do
        for ibv_hw_rate_limit in auto disabled; do
            for mlx5dv in auto disabled; do
                for pcap in auto disabled; do
                    meson configure -Dpython=$python -Dibv=$ibv -Dibv_hw_rate_limit=$ibv_hw_rate_limit -Dmlx5dv=${mlx5dv} -Dpcap=$pcap
                    meson compile
                done
            done
        done
    done
    # sendmmsg and gso don't interact with ibv, so don't test them jointly
    for sendmmsg in auto disabled; do
        for gso in auto disabled; do
            meson configure -Dpython=$python -Dsendmmsg=$sendmmsg -Dgso=$gso
            meson compile
        done
    done
done
