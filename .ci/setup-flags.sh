#!/bin/bash

# Produce flags to meson to force certain features. This ensures that the CI
# builds are actually testing all the optional features, and not failing to
# include them because the detection code is broken.

flags=(
    "-Dwerror=true"
    "-Dtools=enabled"
    "-Dpcap=enabled"
    "-Dunit_test=enabled"
)

if [ "$(uname)" = "Linux" ]; then
    flags+=(
        "-Dibv=enabled"
        "-Dmlx5dv=enabled"
        "-Dibv_hw_rate_limit=enabled"
        "-Dpcap=enabled"
        "-Dcap=enabled"
        "-Drecvmmsg=enabled"
        "-Dsendmmsg=enabled"
        "-Dgso=enabled"
        "-Dgro=enabled"
        "-Dpthread_setaffinity_np=enabled"
        "-Dposix_semaphores=enabled"
        "-Deventfd=enabled"
    )
else
    flags+=("--native-file=ci.ini")
fi

case "$(arch)" in
    x86_64)
        flags+=(
            "-Dsse2_stream=enabled"
            "-Davx_stream=enabled"
            "-Davx512_stream=enabled"
        )
        ;;
    aarch64)
        # Note: Apple uses "arm64" while Linux uses "aarch64". Apple doesn't
        # seem to support SVE in any hardware (up to M4) and our detection
        # code is Linux-specific, so we don't try to force this for MacOS
        # builds.
        flags+=("-Dsve_stream=enabled")
        ;;
esac

echo "Setting flags ${flags[*]}" 1>&2

if [ "$1" = "--python" ]; then
    for arg in "${flags[@]}"; do
        echo -n "--config-settings=setup-args=$arg "
    done
    echo
else
    echo "${flags[*]}"
fi
