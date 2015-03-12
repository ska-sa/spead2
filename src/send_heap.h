/**
 * @file
 */

#ifndef SPEAD_SEND_HEAP_H
#define SPEAD_SEND_HEAP_H

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include "common_defines.h"

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
    /**
     * If true, the item's value is stored in-place and @em must be
     * encoded as an immediate. Non-inline values can still be encoded
     * as immediates if they have the right length.
     */
    bool is_inline;
    union
    {
        struct
        {
            /// Pointer to the value
            const std::uint8_t *ptr;
            /// Length of the value
            std::size_t length;
        } buffer;

        std::uint64_t immediate; ///< Integer value to store (host endian)
    } data;

    item() = default;
    item(std::int64_t id, const void *ptr, std::size_t length)
        : id(id), is_inline(false)
    {
        data.buffer.ptr = reinterpret_cast<const std::uint8_t *>(ptr);
        data.buffer.length = length;
    }

    item(std::int64_t id, std::size_t immediate)
        : id(id), is_inline(true)
    {
        data.immediate = immediate;
    }

    item(std::int64_t id, const std::string &value)
        : item(id, value.data(), value.size())
    {
    }

    item(std::int64_t id, const std::vector<std::uint8_t> &value)
        : item(id, value.data(), value.size())
    {
    }
};

/**
 * Heap with item descriptors pre-baked to items.
 */
class basic_heap
{
    friend class packet_generator;
private:
    std::int64_t cnt;
    std::vector<item> items; ///< Items to write (including descriptors)
    /**
     * Transient storage that should be freed when the heap is no longer
     * needed. Items may point to either this storage or external storage.
     */
    std::vector<std::unique_ptr<std::uint8_t[]> > storage;

public:
    template<typename T>
    basic_heap(std::int64_t cnt, T &&items,
               std::vector<std::unique_ptr<std::uint8_t[]> > &&storage)
    : cnt(cnt), items(std::forward<T>(items)), storage(std::move(storage))
    {
    }

    std::int64_t get_cnt() const
    {
        return cnt;
    }

    void set_cnt(std::int64_t cnt)
    {
        this->cnt = cnt;
    }
};

class heap
{
private:
    std::int64_t cnt;
    bug_compat_mask bug_compat;

    std::vector<item> items;
    std::vector<descriptor> descriptors;

    // Prevent copying
    heap(const heap &) = delete;
    heap &operator=(const heap &) = delete;

public:
    explicit heap(std::int64_t cnt = 0, bug_compat_mask bug_compat = 0)
    : cnt(cnt), bug_compat(bug_compat)
    {
    }

    std::int64_t get_cnt() const
    {
        return cnt;
    }

    void set_cnt(std::int64_t cnt)
    {
        this->cnt = cnt;
    }

    bug_compat_mask get_bug_compat() const
    {
        return bug_compat;
    }

    void set_bug_compat(bug_compat_mask bug_compat)
    {
        this->bug_compat = bug_compat;
    }

    template<typename... Args>
    void add_item(std::int64_t id, Args&&... args)
    {
        items.emplace_back(id, std::forward<Args>(args)...);
    }

    void add_descriptor(const descriptor &descriptor)
    {
        descriptors.push_back(descriptor);
    }

    basic_heap encode(int heap_address_bits) const;
};

} // namespace send
} // namespace spead

#endif // SPEAD_SEND_HEAP_H
