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


version_py = configure_file(
  input : '_version.py.in',
  output : '_version.py',
  configuration: {'VERSION': meson.project_version()}
)
py.install_sources(
  '__init__.py',
  '__init__.pyi',
  'numba.py',
  'numba.pyi',
  'py.typed',
  version_py,
  subdir : 'spead2'
)
subdir('recv')
subdir('send')
subdir('tools')
