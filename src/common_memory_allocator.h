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

#ifndef SPEAD2_COMMON_MEMORY_ALLOCATOR_H
#define SPEAD2_COMMON_MEMORY_ALLOCATOR_H

#include <memory>
#include <cstdint>
#include <cstddef>
#include <utility>

namespace spead2
{

/**
 * Polymorphic class for managing memory allocations in a memory pool. This
 * can be overloaded to provide custom memory allocations.
 */
class memory_allocator : public std::enable_shared_from_this<memory_allocator>
{
public:
    class deleter
    {
    public:
        std::shared_ptr<memory_allocator> allocator;
        void *user = nullptr;

        deleter() = default;
        explicit deleter(std::shared_ptr<memory_allocator> allocator, void *user = nullptr)
            : allocator(std::move(allocator)), user(user) {}

        void operator()(std::uint8_t *ptr);
    };

    typedef std::unique_ptr<std::uint8_t[], deleter> pointer;

    virtual ~memory_allocator() = default;

    /**
     * Allocate @a size bytes of memory. The default implementation uses @c new
     * and pre-faults the memory.
     *
     * @param size         Number of bytes to allocate
     * @returns Pointer to newly allocated memory
     * @throw std::bad_alloc if allocation failed
     */
    virtual pointer allocate(std::size_t size);

private:
    /**
     * Free memory previously returned from @ref allocate.
     *
     * @param ptr          Value returned by @ref allocate
     * @param user         User-defined handle returned by @ref allocate
     */
    virtual void free(std::uint8_t *ptr, void *user);
};

} // namespace spead2

#endif // SPEAD2_COMMON_MEMORY_ALLOCATOR_H
