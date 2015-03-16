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
    s_item_pointer_t id;
    /**
     * If true, the item's value is stored in-place and @em must be
     * encoded as an immediate. Non-inline values can still be encoded
     * as immediates if they have the right length.
     */
    bool is_inline;
    /**
     * If true, the item's value may be encoded as an immediate. This must
     * be false if the item is variable-sized, because in that case the
     * actual size can only be determined from address differences.
     *
     * If @ref is_inline is true, then this must be true as well.
     */
    bool allow_immediate;

    union
    {
        struct
        {
            /// Pointer to the value
            const std::uint8_t *ptr;
            /// Length of the value
            std::size_t length;
        } buffer;

        /**
         * Integer value to store (host endian). This is used if and only
         * if @ref is_inline is true.
         */
        s_item_pointer_t immediate;
    } data;

    item() = default;
    item(s_item_pointer_t id, const void *ptr, std::size_t length, bool allow_immediate)
        : id(id), is_inline(false), allow_immediate(allow_immediate)
    {
        data.buffer.ptr = reinterpret_cast<const std::uint8_t *>(ptr);
        data.buffer.length = length;
    }

    item(s_item_pointer_t id, s_item_pointer_t immediate)
        : id(id), is_inline(true), allow_immediate(true)
    {
        data.immediate = immediate;
    }

    item(s_item_pointer_t id, const std::string &value, bool allow_immediate)
        : item(id, value.data(), value.size(), allow_immediate)
    {
    }

    item(s_item_pointer_t id, const std::vector<std::uint8_t> &value, bool allow_immediate)
        : item(id, value.data(), value.size(), allow_immediate)
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
    s_item_pointer_t cnt;
    std::vector<item> items; ///< Items to write (including descriptors)
    /**
     * Transient storage that should be freed when the heap is no longer
     * needed. Items may point to either this storage or external storage.
     */
    std::vector<std::unique_ptr<std::uint8_t[]> > storage;

public:
    template<typename T>
    basic_heap(s_item_pointer_t cnt, T &&items,
               std::vector<std::unique_ptr<std::uint8_t[]> > &&storage)
    : cnt(cnt), items(std::forward<T>(items)), storage(std::move(storage))
    {
    }

    s_item_pointer_t get_cnt() const
    {
        return cnt;
    }

    void set_cnt(s_item_pointer_t cnt)
    {
        this->cnt = cnt;
    }
};

class heap
{
private:
    s_item_pointer_t cnt;
    bug_compat_mask bug_compat;

    std::vector<item> items;
    std::vector<descriptor> descriptors;

    // Prevent copying
    heap(const heap &) = delete;
    heap &operator=(const heap &) = delete;

public:
    explicit heap(s_item_pointer_t cnt = 0, bug_compat_mask bug_compat = 0)
    : cnt(cnt), bug_compat(bug_compat)
    {
    }

    s_item_pointer_t get_cnt() const
    {
        return cnt;
    }

    void set_cnt(s_item_pointer_t cnt)
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
    void add_item(s_item_pointer_t id, Args&&... args)
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
