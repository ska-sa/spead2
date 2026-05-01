# syntax = docker/dockerfile:1

################################################################################
# Copyright 2024 National Research Foundation (SARAO)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################################

FROM ubuntu:24.04

WORKDIR /tmp/spead2
COPY install-sys-pkgs.sh .

# Install dependencies
# jq and iproute2 are used to find the interface IPv4 address
RUN apt-get update && apt-get -y --no-install-recommends install libpython3-dev python3-venv jq iproute2
RUN CXX=g++ ./install-sys-pkgs.sh

# Give setpriv the ability to run commands with CAP_NET_RAW capability
RUN setcap cap_net_raw+p /usr/bin/setpriv
