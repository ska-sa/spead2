/* Copyright 2016 SKA South Africa
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
 */

#include "common_memory_pool.h"

namespace spead2
{

void memory_allocator::deleter::operator()(std::uint8_t *ptr)
{
    allocator->free(ptr, user);
}

memory_allocator::pointer memory_allocator::allocate(std::size_t size)
{
    std::uint8_t *ptr = new std::uint8_t[size];
    // Pre-fault the memory by touching every page
    for (std::size_t i = 0; i < size; i += 4096)
        ptr[i] = 0;
    return pointer(ptr, deleter(shared_from_this()));
}

void memory_allocator::free(std::uint8_t *ptr, void *user)
{
    (void) user; // prevent warnings about unused parameters
    delete[] ptr;
}

} // namespace spead2
