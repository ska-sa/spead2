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

void memory_allocator::deleter::operator()(pointer ptr)
{
    ptr.get_allocator()->free(ptr.get(), ptr.get_user());
}

void memory_allocator::prefault(std::uint8_t *data, std::size_t size)
{
    // Pre-fault the memory by touching every page
    for (std::size_t i = 0; i < size; i += 4096)
        data[i] = 0;
}

memory_allocator::pointer memory_allocator::allocate(std::size_t size, void *hint)
{
    (void) hint;
    std::uint8_t *ptr = new std::uint8_t[size];
    prefault(ptr, size);
    return pointer(deleter::pointer(ptr, shared_from_this()));
}

void memory_allocator::free(std::uint8_t *ptr, void *user)
{
    (void) user; // prevent warnings about unused parameters
    delete[] ptr;
}

/////////////////////////////////////////////////////////////////////////////

#include <sys/mman.h>

mmap_allocator::mmap_allocator(int flags) : flags(flags)
{
    flags |= MAP_ANONYMOUS | MAP_PRIVATE;
#ifdef MAP_POPULATE
    flags |= MAP_POPULATE;
#endif
}

mmap_allocator::pointer mmap_allocator::allocate(std::size_t size, void *hint)
{
    (void) hint;
    int use_flags = flags | MAP_ANONYMOUS | MAP_PRIVATE
#ifdef MAP_POPULATE
        | MAP_POPULATE
#endif
    ;

    std::uint8_t *ptr = (std::uint8_t *) mmap(nullptr, size, PROT_READ | PROT_WRITE, use_flags, -1, 0);
    if (ptr == MAP_FAILED)
        throw std::bad_alloc();
#ifndef MAP_POPULATE
    prefault(ptr, size);
#endif
    return pointer(deleter::pointer(ptr, shared_from_this(), (void *) std::uintptr_t(size)));
}

void mmap_allocator::free(std::uint8_t *ptr, void *user)
{
    std::size_t size = std::uintptr_t(user);
    munmap(ptr, size);
}

} // namespace spead2
