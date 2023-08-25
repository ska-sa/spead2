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

foreach name : [
  'recv_chunk_example',
  'recv_chunk_ring_example',
  'recv_chunk_group_example',
  'test_recv',
  'test_send',
]
  executable(name, name + '.cpp', dependencies : [st_dep])
endforeach

executable('test_ringbuffer', 'test_ringbuffer.cpp', dependencies : [st_dep, boost_program_options_dep])
if use_ibv
  executable('gpudirect_example', 'gpudirect_example.cu', dependencies : [st_dep, cuda_dep])
  executable('gdrapi_example', 'gdrapi_example.cu', dependencies : [st_dep, cuda_dep, gdrapi_dep])
endif
