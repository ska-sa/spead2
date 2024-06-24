#!/usr/bin/env python3

# Copyright 2023 National Research Foundation (SARAO)
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

import numpy as np

import spead2.recv


def main():
    thread_pool = spead2.ThreadPool()
    stream = spead2.recv.Stream(thread_pool)
    stream.add_udp_reader(8888)
    item_group = spead2.ItemGroup()
    for heap in stream:
        item_group.update(heap)
        timestamp = item_group["timestamp"].value
        power = np.mean(np.square(item_group["adc_samples"].value, dtype=int))
        print(f"Timestamp: {timestamp:<10} Power: {power:.2f}")


if __name__ == "__main__":
    main()
