/* Copyright 2015, 2019 SKA South Africa
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

#ifndef SPEAD2_SEND_HEAP_H
#define SPEAD2_SEND_HEAP_H

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <spead2/common_defines.h>
#include <spead2/common_flavour.h>

namespace spead2
{
namespace send
{

class packet_generator;

/**
 * An item to be inserted into a heap. An item does *not* own its memory.
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

    /**
     * Default constructor. This item has undefined values and is not usable.
     */
    item() = default;

    /**
     * Create an item referencing existing memory.
     */
    item(s_item_pointer_t id, const void *ptr, std::size_t length, bool allow_immediate)
        : id(id), is_inline(false), allow_immediate(allow_immediate)
    {
        data.buffer.ptr = reinterpret_cast<const std::uint8_t *>(ptr);
        data.buffer.length = length;
    }

    /**
     * Create an item with a value to be encoded as an immediate.
     */
    item(s_item_pointer_t id, s_item_pointer_t immediate)
        : id(id), is_inline(true), allow_immediate(true)
    {
        data.immediate = immediate;
    }

    /**
     * Construct an item referencing the data in a string.
     */
    item(s_item_pointer_t id, const std::string &value, bool allow_immediate)
        : item(id, value.data(), value.size(), allow_immediate)
    {
    }

    /**
     * Construct an item referencing the data in a vector.
     */
    item(s_item_pointer_t id, const std::vector<std::uint8_t> &value, bool allow_immediate)
        : item(id, value.data(), value.size(), allow_immediate)
    {
    }
};

/**
 * Heap that is constructed for transmission.
 */
class heap
{
    friend class packet_generator;
private:
    flavour flavour_;
    bool repeat_pointers = false;

    /// Items to write (including descriptors)
    std::vector<item> items;
    /**
     * Transient storage that should be freed when the heap is no longer
     * needed. Items may point to either this storage or external storage.
     */
    std::vector<std::unique_ptr<std::uint8_t[]> > storage;

public:
    /// Opaque handle type for retrieving previously added items.
    typedef std::vector<item>::size_type item_handle;

    /**
     * Constructor.
     *
     * @param flavour_    SPEAD flavour that will be used to encode the heap
     */
    explicit heap(
        const flavour &flavour_ = flavour());

    /// Return flavour
    const flavour &get_flavour() const
    {
        return flavour_;
    }

    /**
     * Construct a new item.
     *
     * @return A handle that can be passed to @ref get_item to update the item
     */
    template<typename... Args>
    item_handle add_item(s_item_pointer_t id, Args&&... args)
    {
        item_handle handle = items.size();
        items.emplace_back(id, std::forward<Args>(args)...);
        return handle;
    }

    /**
     * Get a reference to a previously added item.
     *
     * The retrieved item reference may be modified to update the heap in
     * place. Behaviour is undefined if @a handle is not a handle previously
     * returned by @ref add_item.
     *
     * @param handle   Item handle previously returned from @ref add_item
     */
    item &get_item(item_handle handle)
    {
        return items[handle];
    }

    /**
     * Get a reference to a previously added item.
     *
     * Behaviour is undefined if @a handle is not a handle previously returned
     * by @ref add_item.
     *
     * @param handle   Item handle previously returned from @ref add_item
     */
    const item &get_item(item_handle handle) const
    {
        return items[handle];
    }

    /**
     * Take over ownership of @a pointer and arrange for it to be freed when
     * the heap is freed.
     */
    void add_pointer(std::unique_ptr<std::uint8_t[]> &&pointer)
    {
        storage.push_back(std::move(pointer));
    }

    /**
     * Encode a descriptor to an item and add it to the heap.
     */
    void add_descriptor(const descriptor &descriptor);

    /**
     * Add a start-of-stream control item.
     */
    void add_start()
    {
        add_item(STREAM_CTRL_ID, CTRL_STREAM_START);
    }

    /**
     * Add an end-of-stream control item.
     */
    void add_end()
    {
        add_item(STREAM_CTRL_ID, CTRL_STREAM_STOP);
    }

    /**
     * Enable/disable repetition of item pointers in all packets.
     *
     * Usually this is not needed, but it can enable some specialised use
     * cases where immediates can be recovered from incomplete heaps or where
     * the receiver examines the item pointers in each packet to decide how
     * to handle it. The packet size must be large enough to fit all the item
     * pointers for the heap (the implementation also reserves a little space,
     * so do not rely on a tight fit working).
     *
     * The default is disabled.
     */
    void set_repeat_pointers(bool repeat)
    {
        repeat_pointers = repeat;
    }

    /// Return the flag set by @ref set_repeat_pointers.
    bool get_repeat_pointers() const
    {
        return repeat_pointers;
    }
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_HEAP_H
