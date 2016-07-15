# Copyright 2016 SKA South Africa
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

# SPEAD2_ARG_WITH(ARG, HELP, VAR, TEST)
# ----------------------------------------
# 
AC_DEFUN([SPEAD2_ARG_WITH],
         [AS_VAR_PUSHDEF(var, [$3])
          AS_VAR_PUSHDEF(wvar, [with_$1])
          AS_VAR_SET([var], [0])
          AC_ARG_WITH([$1], [$2], [], [AS_VAR_SET([wvar], [check])])
          AS_VAR_IF([wvar], [no], [],
              [$4
               AS_VAR_IF([wvar], [check], [],
                   [AS_VAR_IF([var], [0],
                        [AC_MSG_ERROR([--with-$1 was specified but not found])])])
              ])
          AC_SUBST($3)
          AS_VAR_POPDEF([var])
          AS_VAR_POPDEF([wvar])
        ])
