/* Copyright 2016 National Research Foundation (SARAO)
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

#ifndef SPEAD2_COMMON_MEMCPY_H
#define SPEAD2_COMMON_MEMCPY_H

#include <cstddef>
#include <spead2/common_features.h>
#include <spead2/common_defines.h>

/**
 * Variant of memcpy that uses a non-temporal hint for the destination.
 * This is not necessarily any faster on its own (and may be slower), but it
 * avoids polluting the cache.
 *
 * If compiler and run-time support is not available, this falls back to
 * regular memcpy.
 *
 * On AArch64, this does not carry a dependency from the source address to
 * the source data. In other words, if you do the following, proper ordering
 * is not guaranteed:
 *
 * 1. Write data to an array;
 * 2. Write the address of the array to an atomic pointer;
 * 3. In another thread, read the pointer with @c std::memory_order_consume and
 *    pass it to this function.
 *
 * Rather use @c std::memory_order_acquire in this case.
 */
namespace spead2
{

void *memcpy_nontemporal(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept;

} // namespace spead2

#endif // SPEAD2_COMMON_MEMCPY_H
