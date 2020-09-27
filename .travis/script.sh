#!/bin/bash
set -e -v

set +v
source venv/bin/activate
set -v
./bootstrap.sh

if [ "$TEST_CXX" = "yes" ]; then
    mkdir -p build
    pushd build
    ../configure \
        --with-recvmmsg="${RECVMMSG:-no}" \
        --with-sendmmsg="${SENDMMSG:-no}" \
        --with-eventfd="${EVENTFD:-no}" \
        --with-ibv="${IBV:-no}" \
        --with-ibv-hw-rate-limit="${IBV_HW_RATE_LIMIT:-no}" \
        --with-mlx5dv="${MLX5DV:-no}" \
        --with-pcap="${PCAP:-no}" \
        --enable-coverage="${COVERAGE:-no}" \
        --disable-optimized \
        CXXFLAGS=-Werror
    make -j4
    if ! make -j4 check; then
        cat src/test-suite.log
        exit 1
    fi
    popd
fi

if [ "$TEST_PYTHON" = "yes" ]; then
    if [ "$COVERAGE" = "yes" ]; then
        echo '[build_ext]' > setup.cfg
        echo 'coverage = yes' >> setup.cfg
        # pip's build isolation prevents us getting .gcno files, so build in place
        CC="$CC -Werror" pip install -v -e .
    else
        CC="$CC -Werror" pip install -v .
    fi
    pytest
    for test in test_logging_shutdown test_running_thread_pool test_running_stream; do
        echo "Running shutdown test $test"
        python -c "import tests.shutdown; tests.shutdown.$test()"
    done
    flake8
fi
