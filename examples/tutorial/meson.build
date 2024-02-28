# Copyright 2023-2024 National Research Foundation (SARAO)
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

foreach name : [
  'tut_2_send',
  'tut_3_recv',
  'tut_4_send_perf',
  'tut_5_send_pipeline',
  'tut_6_send_pktsize',
  'tut_7_recv_power',
  'tut_8_send_reuse_memory',
  'tut_9_recv_memory_pool',
  'tut_10_send_reuse_heaps',
  'tut_11_send_batch_heaps',
  'tut_12_recv_chunks',
]
  executable(name, name + '.cpp', dependencies : [st_dep])
endforeach
