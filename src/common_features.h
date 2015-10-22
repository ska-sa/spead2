/* Copyright 2015 SKA South Africa
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 *
 * Macros to control optional features
 */

#ifndef SPEAD2_COMMON_FEATURES_H
#define SPEAD2_COMMON_FEATURES_H

// Include some glibc header to get __GLIBC_PREREQ
#include <climits>

// recvmmsg support was added to glibc 2.12
#ifndef SPEAD2_USE_RECVMMSG
# if defined(__APPLE__)
#  define SPEAD2_USE_RECVMMSG 0
# elif defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 12)
#  define SPEAD2_USE_RECVMMSG 1
# else
#  define SPEAD2_USE_RECVMMSG 0
# endif
#endif

#endif // SPEAD2_COMMON_FEATURES_H
