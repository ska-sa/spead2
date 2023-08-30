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

header_install_dir = get_option('includedir') / 'spead2'

common_features_h = configure_file(
  input : 'common_features.h.in',
  output : 'common_features.h',
  configuration : conf,
  install : true,
  install_dir : header_install_dir
)

gen_headers = [common_features_h]
foreach name : loader_modules
  h_name = f'common_loader_@name@.h'
  gen_h = custom_target(
    input : [gen_loader],
    output : [h_name],
    depend_files : gen_templates,
    command : [py, '@INPUT@', 'header', name, '@OUTPUT@'],
    install : true,
    install_dir : header_install_dir
  )
  gen_headers += gen_h
endforeach

install_subdir(
  '.',
  install_dir : header_install_dir,
  exclude_files : ['common_features.h.in', 'meson.build'],
  install_tag : 'devel'
)
