/**
 * @file
 */

#ifndef SPEAD_COMMON_MEMPOOL_H
#define SPEAD_COMMON_MEMPOOL_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stack>
#include <memory>

namespace spead
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
 * This class is thread-safe.
 */
class mempool
{
public:
    typedef std::unique_ptr<std::uint8_t[], std::function<void(std::uint8_t *)> > pointer;

private:
    std::size_t lower, upper, max_free;
    std::mutex mutex;
    std::stack<std::unique_ptr<std::uint8_t[]> > pool;

    void return_to_pool(std::uint8_t *ptr);
    std::unique_ptr<std::uint8_t[]> allocate_for_pool();

public:
    mempool();
    mempool(std::size_t lower, std::size_t upper, std::size_t max_free, std::size_t initial);
    pointer allocate(std::size_t size);
};

} // namespace spead

#endif // SPEAD_COMMON_MEMPOOL_H
