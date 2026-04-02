if [ "$(uname -s)" = "Linux" ]; then
    SUDO=sudo
    if [ `id -u` -eq 0 ]; then
        SUDO=  # We're already root, and sudo might not exist
    fi
    $SUDO apt-get update && $SUDO apt-get -y --no-install-recommends install \
        python3-all \
        python3-venv \
        libpython3-dev
    python3 -m venv .venv
else
    brew update
    brew install python3 python3-venv libpython3-dev
fi
