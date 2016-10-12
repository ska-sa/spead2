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

## Check whether a library is available. Unlike AC_CHECK_LIB, this supports
## C++. It takes a header file, a library to link against, and a code snippet
## to build inside main().
## Unlike other autoconf tests, action-if-found defaults to nothing

#serial 1

AC_DEFUN([_SPEAD2_INCLUDE], [#include <$1>
])

# SPEAD2_CHECK_FEATURE(NAME, DESCRIPTION, HEADERS, LIBRARY, BODY,
#                      [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND], [EXTRA-PREAMBLE])
# -------------------------------------------------------------------------------
AC_DEFUN([SPEAD2_CHECK_FEATURE], [
    AS_VAR_PUSHDEF([cv], [spead2_cv_$1])
    AC_CACHE_CHECK([for $2], [cv], [
        spead2_save_LIBS=$LIBS
        m4_ifval([$4], [LIBS="m4_foreach([i], [$4], [[-l]i ])$LIBS"],)
        AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([$8
                              m4_map_args_w([$3], [_SPEAD2_INCLUDE(], [)])], [$5])],
            [AS_VAR_SET([cv], [yes])],
            [AS_VAR_SET([cv], [no])])
        LIBS=$spead2_save_LIBS
    ])
    AS_VAR_IF([cv], [yes], [$6], [$7])
    AS_VAR_POPDEF([cv])
])

# SPEAD2_CHECK_LIB(HEADERS, LIBRARY, BODY, ACTION-IF-FOUND, ACTION-IF-NOT-FOUND)
# -----------------------------------------------------------------------------
AC_DEFUN([SPEAD2_CHECK_LIB], [SPEAD2_CHECK_FEATURE([lib_$2], [-l$2], [$1], [$2], [$3], [$4], [$5])])

# SPEAD2_CHECK_HEADER(HEADER, LIBRARY, BODY, ACTION-IF-FOUND, ACTION-IF-NOT-FOUND)
# -----------------------------------------------------------------------------
AC_DEFUN([SPEAD2_CHECK_HEADER], [SPEAD2_CHECK_FEATURE([header_$1], [$1], [$1], [$2], [$3],[$4], [$5])])
