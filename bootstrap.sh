#!/bin/bash
set -e

python3 gen/gen_loader.py header rdmacm > include/spead2/common_loader_rdmacm.h
python3 gen/gen_loader.py header ibv > include/spead2/common_loader_ibv.h
python3 gen/gen_loader.py header mlx5dv > include/spead2/common_loader_mlx5dv.h
python3 gen/gen_loader.py cxx rdmacm > src/common_loader_rdmacm.cpp
python3 gen/gen_loader.py cxx ibv > src/common_loader_ibv.cpp
python3 gen/gen_loader.py cxx mlx5dv > src/common_loader_mlx5dv.cpp
autoreconf --install --force
