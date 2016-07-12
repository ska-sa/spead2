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
#include <cstdint>
#include <spead2/common_memory_pool.h>
#include <spead2/common_logging.h>

namespace spead2
{

memory_pool::memory_pool()
    : memory_pool(nullptr, 0, 0, 0, 0, 0, nullptr)
{
}

memory_pool::memory_pool(std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
                         std::shared_ptr<memory_allocator> allocator)
    : memory_pool(nullptr, lower, upper, max_free, initial, 0, std::move(allocator))
{
}

memory_pool::memory_pool(
    boost::asio::io_service &io_service,
    std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
    std::size_t low_water,
    std::shared_ptr<memory_allocator> allocator)
    : memory_pool(&io_service, lower, upper, max_free, initial, low_water, std::move(allocator))
{
}

memory_pool::memory_pool(
    thread_pool &tpool,
    std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
    std::size_t low_water,
    std::shared_ptr<memory_allocator> allocator)
    : memory_pool(&tpool.get_io_service(), lower, upper, max_free, initial, low_water,
                  std::move(allocator))
{
}

memory_pool::memory_pool(
    boost::asio::io_service *io_service,
    std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
    std::size_t low_water,
    std::shared_ptr<memory_allocator> allocator)
    : io_service(io_service), lower(lower), upper(upper), max_free(max_free),
    initial(initial), low_water(low_water),
    base_allocator(allocator ? move(allocator) : std::make_shared<memory_allocator>())
{
    assert(lower <= upper);
    assert(initial <= max_free);
    assert(low_water <= initial);
    assert(low_water == 0 || io_service != nullptr);
    for (std::size_t i = 0; i < initial; i++)
        pool.emplace(base_allocator->allocate(upper, nullptr));
}

void memory_pool::free(std::uint8_t *ptr, void *user)
{
    pointer wrapped(ptr, deleter(base_allocator, user));
    std::unique_lock<std::mutex> lock(mutex);
    if (pool.size() < max_free)
    {
        log_debug("returning memory to the pool");
        pool.push(move(wrapped));
    }
    else
    {
        log_debug("dropping memory because the pool is full");
        lock.unlock();
        // deleter for wrapped will free the memory
    }
}

memory_pool::pointer memory_pool::convert(pointer &&base)
{
    pointer wrapped(base.get(), deleter(shared_from_this(), base.get_deleter().get_user()));
    base.release();
    return wrapped;
}

void memory_pool::refill(std::size_t upper, std::shared_ptr<memory_allocator> allocator,
                         std::weak_ptr<memory_pool> self_weak)
{
    while (true)
    {
        pointer ptr = allocator->allocate(upper, nullptr);
        std::shared_ptr<memory_pool> self = self_weak.lock();
        if (!self)
        {
            break;  // The memory pool vanished from under us
        }
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

memory_pool::pointer memory_pool::allocate(std::size_t size, void *hint)
{
    (void) hint;
    pointer ptr;
    if (size >= lower && size <= upper)
    {
        /* Declaration order here is important: if there is an exception,
         * we want to drop the lock before trying to put the pointer back in
         * the pool.
         */
        std::unique_lock<std::mutex> lock(mutex);
        if (!pool.empty())
        {
            ptr = std::move(pool.top());
            pool.pop();
            ptr = convert(std::move(ptr));
            if (pool.size() < low_water && !refilling)
            {
                refilling = true;
                std::shared_ptr<memory_pool> self =
                    std::static_pointer_cast<memory_pool>(shared_from_this());
                std::weak_ptr<memory_pool> weak{self};
                // C++ (or at least GCC) won't let me capture the members by value directly
                const std::size_t upper = this->upper;
                std::shared_ptr<memory_allocator> allocator = base_allocator;
                io_service->post([upper, allocator, weak] {
                    refill(upper, allocator, std::move(weak));
                });
            }
            lock.unlock();
            log_debug("allocating %d bytes from pool", size);
        }
        else
        {
            lock.unlock();
            ptr = convert(base_allocator->allocate(upper, nullptr));
            log_debug("allocating %d bytes which will be added to the pool", size);
        }
    }
    else
    {
        log_debug("allocating %d bytes without using the pool", size);
        ptr = base_allocator->allocate(size, nullptr);
    }
    return ptr;
}

} // namespace spead2
