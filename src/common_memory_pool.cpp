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
 */

#include <cassert>
#include <utility>
#include <memory>
#include "common_memory_pool.h"
#include "common_logging.h"

namespace spead2
{

memory_pool::memory_pool()
    : memory_pool(nullptr, 0, 0, 0, 0, 0)
{
}

memory_pool::memory_pool(std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial)
    : memory_pool(nullptr, lower, upper, max_free, initial, 0)
{
}

memory_pool::memory_pool(
    boost::asio::io_service &io_service,
    std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
    std::size_t low_water)
    : memory_pool(&io_service, lower, upper, max_free, initial, low_water)
{
}

memory_pool::memory_pool(
    thread_pool &tpool,
    std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
    std::size_t low_water)
    : memory_pool(&tpool.get_io_service(), lower, upper, max_free, initial, low_water)
{
}

memory_pool::memory_pool(
    boost::asio::io_service *io_service,
    std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
    std::size_t low_water)
    : io_service(io_service), lower(lower), upper(upper), max_free(max_free),
    initial(initial), low_water(low_water)
{
    assert(lower <= upper);
    assert(initial <= max_free);
    assert(low_water <= initial);
    assert(low_water == 0 || io_service != nullptr);
    for (std::size_t i = 0; i < initial; i++)
        pool.emplace(allocate_for_pool(upper));
}

memory_pool::destructor::destructor(std::shared_ptr<memory_pool> owner)
    : owner(std::move(owner))
{
}

void memory_pool::destructor::operator()(std::uint8_t *ptr) const
{
    if (owner)
        owner->return_to_pool(ptr);
    else
    {
        // Not allocated from pool
        delete[] ptr;
    }
}

void memory_pool::return_to_pool(std::uint8_t *ptr)
{
    std::unique_lock<std::mutex> lock(mutex);
    if (pool.size() < max_free)
    {
        log_debug("returning memory to the pool");
        pool.emplace(ptr);
    }
    else
    {
        log_debug("dropping memory because the pool is full");
        lock.unlock();
        delete[] ptr;
    }
}

std::unique_ptr<std::uint8_t[]> memory_pool::allocate_for_pool(std::size_t upper)
{
    std::uint8_t *ptr = new std::uint8_t[upper];
    // Pre-fault the memory by touching every page
    for (std::size_t i = 0; i < upper; i += 4096)
        ptr[i] = 0;
    return std::unique_ptr<std::uint8_t[]>(ptr);
}

void memory_pool::refill(std::size_t upper, std::weak_ptr<memory_pool> self_weak)
{
    while (true)
    {
        std::unique_ptr<std::uint8_t[]> ptr = allocate_for_pool(upper);
        std::shared_ptr<memory_pool> self = self_weak.lock();
        if (!self)
            break;  // The memory pool vanished from under us
        std::lock_guard<std::mutex> lock(self->mutex);
        if (self->pool.size() < self->max_free)
        {
            log_debug("adding background memory to the pool");
            self->pool.push(std::move(ptr));
        }
        if (self->pool.size() >= self->initial)
        {
            self->refilling = false;
            log_debug("exiting refill task");
            break;
        }
    }
}

memory_pool::pointer memory_pool::allocate(std::size_t size)
{
    if (size >= lower && size <= upper)
    {
        std::unique_lock<std::mutex> lock(mutex);
        std::unique_ptr<uint8_t[]> ptr;
        if (!pool.empty())
        {
            ptr.reset(pool.top().release());
            pool.pop();
            if (pool.size() < low_water && !refilling)
            {
                refilling = true;
                std::weak_ptr<memory_pool> weak{shared_from_this()};
                // C++ (or at least GCC) won't let me capture the member by value directly
                const std::size_t upper = this->upper;
                io_service->post([upper, weak] { refill(upper, std::move(weak)); });
            }
            lock.unlock();
            log_debug("allocating %d bytes from pool", size);
        }
        else
        {
            lock.unlock();
            ptr = allocate_for_pool(upper);
            log_debug("allocating %d bytes which will be added to the pool", size);
        }
        // The pool may be discarded while the memory is still allocated: in
        // this case, it is simply freed.
        return pointer(ptr.release(), destructor(shared_from_this()));
    }
    else
    {
        log_debug("allocating %d bytes without using the pool", size);
        return pointer(new std::uint8_t[size]);
    }
}

} // namespace spead2

#include "common_memory_pool.h"
