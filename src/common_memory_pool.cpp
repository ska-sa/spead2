/* Copyright 2015, 2017, 2021 National Research Foundation (SARAO)
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
    : memory_pool(boost::none, 0, 0, 0, 0, 0, nullptr)
{
}

memory_pool::memory_pool(std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
                         std::shared_ptr<memory_allocator> allocator)
    : memory_pool(boost::none, lower, upper, max_free, initial, 0, std::move(allocator))
{
}

memory_pool::memory_pool(
    io_service_ref io_service,
    std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
    std::size_t low_water,
    std::shared_ptr<memory_allocator> allocator)
    : memory_pool(boost::optional<io_service_ref>(std::move(io_service)),
                  lower, upper, max_free, initial, low_water, std::move(allocator))
{
}

namespace detail
{

class memory_pool_deleter
{
private:
    struct state_t
    {
        std::shared_ptr<memory_pool> allocator;
        memory_allocator::deleter base_deleter;

        state_t(std::shared_ptr<memory_pool> &&allocator,
                memory_allocator::deleter &&base_deleter)
            : allocator(std::move(allocator)), base_deleter(std::move(base_deleter)) {}
    };

    // See the comments in memory_allocator::legacy_deleter for an explanation
    // of why this is wrapped in a shared_ptr.
    std::shared_ptr<state_t> state;

public:
    memory_pool_deleter(
            std::shared_ptr<memory_pool> &&allocator,
            memory_allocator::deleter &&base_deleter)
        : state(std::make_shared<state_t>(std::move(allocator), std::move(base_deleter)))
    {
    }

    void operator()(std::uint8_t *ptr) const
    {
        state->allocator->free_impl(ptr, std::move(state->base_deleter));
        // Allow the allocator to be freed even if the unique_ptr lingers on.
        state->allocator.reset();
    }

    memory_allocator::deleter &get_base_deleter() { return state->base_deleter; }    
    const memory_allocator::deleter &get_base_deleter() const { return state->base_deleter; }
};

} // namespace detail

memory_pool::memory_pool(
    boost::optional<io_service_ref> io_service,
    std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial,
    std::size_t low_water,
    std::shared_ptr<memory_allocator> allocator)
    : io_service(std::move(io_service)), lower(lower), upper(upper), max_free(max_free),
    initial(initial), low_water(low_water),
    base_allocator(allocator ? move(allocator) : std::make_shared<memory_allocator>())
{
    assert(lower <= upper);
    assert(initial <= max_free);
    assert(low_water <= initial);
    assert(low_water == 0 || io_service);
    for (std::size_t i = 0; i < initial; i++)
        pool.emplace(base_allocator->allocate(upper, nullptr));
}

std::shared_ptr<memory_pool> memory_pool::shared_this()
{
    return std::static_pointer_cast<memory_pool>(shared_from_this());
}

void memory_pool::free_impl(std::uint8_t *ptr, memory_allocator::deleter &&base_deleter)
{
    pointer wrapped(ptr, std::move(base_deleter));
    std::unique_lock<std::mutex> lock(mutex);
    if (pool.size() < max_free)
    {
        log_debug("returning memory to the pool");
        pool.push(std::move(wrapped));
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
    /* TODO: in theory this might not be exception-safe, because in C++11
     * the move constructor for std::function is not noexcept. Thus, after
     * constructing the memory_pool_deleter argument, the construction of
     * the pointer could fail. The lack of noexcept is assumed to be an
     * oversight in older C++ standards, and GCC 9 at least makes it
     * no-except.
     */
    pointer wrapped(base.get(),
                    detail::memory_pool_deleter(shared_this(), std::move(base.get_deleter())));
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
                std::shared_ptr<memory_pool> self = shared_this();
                std::weak_ptr<memory_pool> weak{self};
                // C++ (or at least GCC) won't let me capture the members by value directly
                const std::size_t upper = this->upper;
                std::shared_ptr<memory_allocator> allocator = base_allocator;
                (*io_service)->post([upper, allocator, weak] {
                    refill(upper, allocator, std::move(weak));
                });
            }
            lock.unlock();
            log_debug("allocating %d bytes from pool", size);
        }
        else
        {
            // Copy the flag while the lock is held, to safely issue the
            // warning after it is released.
            bool warn = warn_on_empty;
            lock.unlock();
            ptr = convert(base_allocator->allocate(upper, nullptr));
            if (warn)
                log_warning("memory pool is empty when allocating %d bytes", size);
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

void memory_pool::set_warn_on_empty(bool warn)
{
    std::lock_guard<std::mutex> lock(mutex);
    warn_on_empty = warn;
}

bool memory_pool::get_warn_on_empty() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return warn_on_empty;
}

const memory_allocator::deleter &memory_pool::get_base_deleter(const memory_allocator::pointer &ptr)
{
    const memory_allocator::deleter *out = &ptr.get_deleter();
    const detail::memory_pool_deleter *pool_del;
    while ((pool_del = out->target<detail::memory_pool_deleter>()) != nullptr)
        out = &pool_del->get_base_deleter();
    return *out;
}

memory_allocator::deleter &memory_pool::get_base_deleter(memory_allocator::pointer &ptr)
{
    memory_allocator::deleter *out = &ptr.get_deleter();
    detail::memory_pool_deleter *pool_del;
    while ((pool_del = out->target<detail::memory_pool_deleter>()) != nullptr)
        out = &pool_del->get_base_deleter();
    return *out;
}

} // namespace spead2
