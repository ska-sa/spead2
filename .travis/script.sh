#!/bin/bash
set -e -v

set +v
source venv/bin/activate
set -v
./bootstrap.sh

if [ "$NETMAP" = "yes" ]; then
    export CPATH="$PWD/netmap/sys"
fi

if [ "$TEST_CXX" = "yes" ]; then
    mkdir -p build
    pushd build
    ../configure \
        --with-netmap="${NETMAP:-no}" \
        --with-recvmmsg="${RECVMMSG:-no}" \
        --with-sendmmsg="${SENDMMSG:-no}" \
        --with-eventfd="${EVENTFD:-no}" \
        --with-ibv="${IBV:-no}" \
        --with-pcap="${PCAP:-no}" \
        --enable-coverage="${COVERAGE:-no}" \
        --disable-optimized \
        CXXFLAGS=-Werror
    make -j4
    make -j4 check
    popd
fi

if [ "$TEST_PYTHON" = "yes" ]; then
    if [ "$COVERAGE" = "yes" ]; then
        echo '[build_ext]' > setup.cfg
        echo 'coverage = yes' >> setup.cfg
        # pip's build isolation prevents us getting .gcno files, so build with setuptools
        CC="$CC -Werror" python ./setup.py install
    else
        CC="$CC -Werror" pip install -v .
    fi
    # Avoid running nosetests from installation directory, to avoid picking up
    # things from the local tree that aren't installed.
    pushd /
    nosetests --with-ignore-docstring -v spead2
    for test in test_logging_shutdown test_running_thread_pool test_running_stream; do
        echo "Running shutdown test $test"
        python -c "import spead2.test.shutdown; spead2.test.shutdown.$test()"
    done
    popd
    flake8
fi
