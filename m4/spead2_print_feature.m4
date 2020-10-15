# Copyright 2020 National Research Foundation (SARAO)
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

#serial 1

# SPEAD2_PRINT_FEATURE(PREFIX, TEST)
# ---------------------------------
AC_DEFUN([SPEAD2_PRINT_FEATURE],
         [AS_IF([$2], [echo "    $1: yes"], [echo "    $1: no"])])

# SPEAD2_PRINT_CONDITION(PREFIX, CONDITIONAL)
# --------------------------------------
AC_DEFUN([SPEAD2_PRINT_CONDITION],
         [AM_COND_IF([$2], [echo "    $1: yes"], [echo "    $1: no"])])
