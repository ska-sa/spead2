#!/bin/bash
set -e

python gen/gen_ibv_loader.py header > include/spead2/common_ibv_loader.h
python gen/gen_ibv_loader.py cxx > src/common_ibv_loader.cpp
autoreconf --install --force
