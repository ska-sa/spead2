/**
 * @file
 */

#include <cassert>
#include "common_memory_pool.h"
#include "common_logging.h"

namespace spead2
{

memory_pool::memory_pool() : lower(0), upper(0), max_free(0)
{
}

memory_pool::memory_pool(std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial)
    : lower(lower), upper(upper), max_free(max_free)
{
    assert(lower <= upper);
    assert(initial <= max_free);
    for (std::size_t i = 0; i < initial; i++)
        pool.emplace(new std::uint8_t[upper]);
}

void memory_pool::return_to_pool(const std::weak_ptr<memory_pool> &self_weak, std::uint8_t *ptr)
{
    std::shared_ptr<memory_pool> self = self_weak.lock();
    if (self)
        self->return_to_pool(ptr);
    else
    {
        log_debug("dropping memory because the pool has been freed");
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

std::unique_ptr<std::uint8_t[]> memory_pool::allocate_for_pool()
{
    std::uint8_t *ptr = new std::uint8_t[upper];
    // Pre-fault the memory by touching every page
    for (std::size_t i = 0; i < upper; i += 4096)
        ptr[i] = 0;
    return std::unique_ptr<std::uint8_t[]>(ptr);
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
            lock.unlock();
            log_debug("allocating %d bytes from pool", size);
        }
        else
        {
            lock.unlock();
            ptr = allocate_for_pool();
            log_debug("allocating %d bytes which will be added to the pool", size);
        }
        // The pool may be discarded while the memory is still allocated: in
        // this case, it is simply freed.
        std::weak_ptr<memory_pool> self(shared_from_this());
        // this is not actually used (and shouldn't be, because this might have
        // been freed), but a user has reported a compiler error if it is not
        // captured.
        return pointer(ptr.release(), [this, self] (std::uint8_t *p) { return_to_pool(self, p); });
    }
    else
    {
        log_debug("allocating %d bytes without using the pool", size);
        return pointer(new std::uint8_t[size], std::default_delete<std::uint8_t[]>());
    }
}

} // namespace spead2

#include "common_memory_pool.h"
