#!/bin/bash
set -e

python gen/gen_loader.py header rdmacm > include/spead2/common_loader_rdmacm.h
python gen/gen_loader.py header ibv > include/spead2/common_loader_ibv.h
python gen/gen_loader.py cxx rdmacm > src/common_loader_rdmacm.cpp
python gen/gen_loader.py cxx ibv > src/common_loader_ibv.cpp
autoreconf --install --force
