/**
 * @file
 */

#ifndef SPEAD_SEND_HEAP_H
#define SPEAD_SEND_HEAP_H

#include <vector>
#include <utility>
#include <cstddef>
#include <cstdint>

namespace spead
{
namespace send
{

class packet_generator;

/**
 * An item to be inserted into a heap.
 */
struct item
{
    /// Item ID
    std::int64_t id;
    /// Start of memory containing value
    const std::uint8_t *ptr;
    /// Start of memory containing length
    std::size_t length;

    item() = default;
    item(std::int64_t id, const void *ptr, std::size_t length)
        : id(id), ptr(reinterpret_cast<const std::uint8_t *>(ptr)), length(length)
    {
    }
};

class heap
{
    friend class packet_generator;
private:
    std::int64_t heap_cnt;
    std::vector<item> items;

public:
    template<typename T>
    heap(std::int64_t heap_cnt, T &&items)
    : heap_cnt(heap_cnt), items(std::forward<T>(items))
    {
    }
};

} // namespace send
} // namespace spead

#endif // SPEAD_SEND_HEAP_H
