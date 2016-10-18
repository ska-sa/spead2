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

#include <spead2/common_memory_pool.h>

// Some operating systems only provide MAP_ANON
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

namespace spead2
{

memory_allocator::deleter::deleter(std::shared_ptr<memory_allocator> allocator,
                                   void *user)
    : allocator(std::move(allocator)), user(user)
{
}

void memory_allocator::deleter::operator()(std::uint8_t *ptr)
{
    allocator->free(ptr, user);
    allocator.reset();
}

void memory_allocator::prefault(std::uint8_t *data, std::size_t size)
{
    // Pre-fault the memory by touching every page
    for (std::size_t i = 0; i < size; i += 4096)
        data[i] = 0;
}

memory_allocator::pointer memory_allocator::allocate(std::size_t size, void *hint)
{
    (void) hint; // prevent warnings about unused parameters
    std::uint8_t *ptr = new std::uint8_t[size];
    prefault(ptr, size);
    return pointer(ptr, deleter(shared_from_this()));
}

void memory_allocator::free(std::uint8_t *ptr, void *user)
{
    (void) user; // prevent warnings about unused parameters
    delete[] ptr;
}

/////////////////////////////////////////////////////////////////////////////

#include <sys/mman.h>

mmap_allocator::mmap_allocator(int flags, bool prefer_huge)
    : flags(flags), prefer_huge(prefer_huge)
{
}

mmap_allocator::pointer mmap_allocator::allocate(std::size_t size, void *hint)
{
    (void) hint;
    int use_flags = flags | MAP_ANONYMOUS | MAP_PRIVATE
#ifdef MAP_POPULATE
        | MAP_POPULATE
#endif
    ;

    std::uint8_t *ptr = (std::uint8_t *) MAP_FAILED;
#ifdef MAP_HUGETLB
    if (prefer_huge)
        ptr = (std::uint8_t *) mmap(nullptr, size, PROT_READ | PROT_WRITE, use_flags | MAP_HUGETLB, -1, 0);
#endif
    if (ptr == MAP_FAILED)
    {
        // Either fallback from prefer-huge, or prefer_huge is false
        ptr = (std::uint8_t *) mmap(nullptr, size, PROT_READ | PROT_WRITE, use_flags, -1, 0);
    }

    if (ptr == MAP_FAILED)
        throw std::bad_alloc();
#ifndef MAP_POPULATE
    prefault(ptr, size);
#endif
    return pointer(ptr, deleter(shared_from_this(), (void *) std::uintptr_t(size)));
}

void mmap_allocator::free(std::uint8_t *ptr, void *user)
{
    std::size_t size = std::uintptr_t(user);
    munmap(ptr, size);
}

} // namespace spead2
