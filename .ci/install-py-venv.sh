if [ "$(uname -s)" = "Linux" ]; then
    SUDO=sudo
    if [ `id -u` -eq 0 ]; then
        SUDO=  # We're already root, and sudo might not exist
    fi
    export DEBIAN_FRONTEND=noninteractive
    export TZ="${TZ:-Etc/UTC}"
    $SUDO apt-get update && $SUDO apt-get -y --no-install-recommends install \
        python3-all \
        python3-venv \
        libpython3-dev

    if [ -n "$PYTHON_VERSION" ]; then
        # pyenv builds CPython from source, so we need build dependencies.
        $SUDO apt-get -y --no-install-recommends install \
            ca-certificates \
            curl \
            git \
            build-essential \
            libssl-dev \
            zlib1g-dev \
            libbz2-dev \
            libreadline-dev \
            libsqlite3-dev \
            libffi-dev \
            liblzma-dev \
            tk-dev

        # Ensure pyenv is usable in non-interactive shells.
        export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
        export PATH="$PYENV_ROOT/bin:$PATH"
        if [ ! -d "$PYENV_ROOT" ]; then
            git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
        fi
        eval "$(pyenv init -)"

        pyenv install -s "$PYTHON_VERSION"
        pyenv global "$PYTHON_VERSION"
        python -m venv .venv
    else
        python3 -m venv .venv
    fi
else
    brew update
    brew install python3 python3-venv libpython3-dev
fi
