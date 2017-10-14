/* Copyright 2015-2017 SKA South Africa
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

#ifndef SPEAD2_COMMON_MEMORY_POOL_H
#define SPEAD2_COMMON_MEMORY_POOL_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stack>
#include <memory>
#include <boost/asio.hpp>
#include <boost/optional.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/common_memory_allocator.h>

namespace spead2
{

/**
 * Memory allocator that pre-allocates memory and recycles it. This wastes
 * memory but reduces the number of page faults. It has a lower bound and an
 * upper bound. Allocations of less than the lower bound, or more than the
 * upper bound, are satisfied directly with @c new. Allocations in between
 * the bounds are satisfied from the pool. If the pool is exhausted, the size
 * is increased. If the free pool grows beyond a given size, memory is returned
 * to the OS.
 *
 * The memory pool must be managed by a std::shared_ptr. The caller may safely
 * drop its references, even if there is still memory that has been allocated
 * and not yet freed.
 *
 * This class is thread-safe.
 */
class memory_pool : public memory_allocator
{
private:
    boost::optional<io_service_ref> io_service;
    const std::size_t lower, upper, max_free, initial, low_water;
    const std::shared_ptr<memory_allocator> base_allocator;
    mutable std::mutex mutex;
    /// Free pool; these pointers are owned by the base allocator
    std::stack<pointer> pool;
    bool refilling = false;
    bool warn_on_empty = true;

    virtual void free(std::uint8_t *ptr, void *user) override;
    // Makes ourself the owner
    pointer convert(pointer &&base);
    static void refill(std::size_t upper, std::shared_ptr<memory_allocator> allocator,
                       std::weak_ptr<memory_pool> self_weak);

    memory_pool(boost::optional<io_service_ref> io_service, std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial, std::size_t low_water,
                std::shared_ptr<memory_allocator> allocator);

public:
    memory_pool();
    memory_pool(std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
                std::shared_ptr<memory_allocator> allocator = nullptr);
    memory_pool(io_service_ref io_service, std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial, std::size_t low_water,
                std::shared_ptr<memory_allocator> allocator = nullptr);
    bool get_warn_on_empty() const;
    void set_warn_on_empty(bool warn);
    virtual pointer allocate(std::size_t size, void *hint) override;
};

} // namespace spead2

#endif // SPEAD2_COMMON_MEMORY_POOL_H
