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

#ifndef SPEAD2_RECV_HEAP_H
#define SPEAD2_RECV_HEAP_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <spead2/common_defines.h>
#include <spead2/common_flavour.h>
#include <spead2/common_memory_allocator.h>

namespace spead2
{
namespace recv
{

class live_heap;

/**
 * An item extracted from a heap.
 */
struct item
{
    /// Item ID
    s_item_pointer_t id;
    /// Start of memory containing value
    std::uint8_t *ptr;
    /// Length of memory
    std::size_t length;
    /// The immediate interpreted as an integer (undefined if not immediate)
    item_pointer_t immediate_value;
    /// Whether the item is immediate
    bool is_immediate;
};

/**
 * Base class for @ref heap and @ref incomplete_heap
 */
class heap_base
{
private:
    s_item_pointer_t cnt;       ///< Heap ID
    flavour flavour_;           ///< Flavour
    /**
     * Extracted items. The pointers in the items point into either @ref
     * payload, @ref immediate_payload_inline or @ref immediate_payload.
     */
    std::vector<item> items;
    /**@{*/
    /**
     * Storage for immediate values. If the number of items is small enough,
     * they are stored in the object itself, avoiding the cost of a malloc.
     * Otherwise they are stored in dynamically allocated memory. It is not
     * necessary to store an indicator of which storage is used because the
     * storage is accessed via the items.
     */
    std::uint8_t immediate_payload_inline[24];   // 4 items in SPEAD-64-48
    std::unique_ptr<std::uint8_t[]> immediate_payload;
    /**@}*/

    /* Copy inline immediate items and fix up pointers to it */
    void transfer_immediates(heap_base &&other) noexcept;

protected:
    /// Create the structures from a live heap, destroying it in the process.
    void load(live_heap &&h, bool keep_addressed, bool keep_payload);
    /**
     * Heap payload. For an incomplete heap, this might or might not be set,
     * depending on constructor parameters.
     */
    memory_allocator::pointer payload;

public:
    heap_base() = default;
    heap_base(heap_base &&other) noexcept;
    heap_base &operator=(heap_base &&other) noexcept;

    /// Get heap ID
    s_item_pointer_t get_cnt() const { return cnt; }
    /// Get protocol flavour used
    const flavour &get_flavour() const { return flavour_; }
    /**
     * Get the items from the heap. This includes descriptors, but
     * excludes any items with ID <= 4.
     */
    const std::vector<item> &get_items() const { return items; }

    /**
     * Convenience function to check whether any of the items is
     * a @c STREAM_CTRL_ID item with value @a value.
     */
    bool is_ctrl_item(ctrl_mode value) const;

    /**
     * Convenience function to check whether any of the items is
     * a @c CTRL_STREAM_START.
     */
    bool is_start_of_stream() const;

    /**
     * Convenience function to check whether any of the items is
     * a @c CTRL_STREAM_STOP.
     */
    bool is_end_of_stream() const;
};

/**
 * Received heap that has been finalised.
 */
class heap : public heap_base
{
public:
    /**
     * Freeze a heap, which must satisfy live_heap::is_contiguous. The original
     * heap is destroyed.
     */
    explicit heap(live_heap &&h);

    /**
     * Extract descriptor fields from the heap. Any missing fields are
     * default-initialized. This should be used on a heap constructed from
     * the content of a descriptor item.
     *
     * The original PySPEAD package (version 0.5.2) does not follow the
     * specification here. The macros in @ref common_defines.h can be
     * used to control whether to interpret the specification or be
     * bug-compatible.
     *
     * The protocol allows descriptors to use immediate-mode items,
     * but the decoding of these into variable-length strings is undefined.
     * This implementation will discard such descriptor fields.
     */
    descriptor to_descriptor() const;

    /// Extract and decode descriptors from this heap
    std::vector<descriptor> get_descriptors() const;
};

/**
 * Received heap that has been finalised, but which is missing data.
 *
 * The payload and any items that refer to the payload are discarded.
 */
class incomplete_heap : public heap_base
{
private:
    /**
     * Contiguous ranges of the payload that were received.
     *
     * @see @ref live_heap::payload_ranges.
     */
    std::map<s_item_pointer_t, s_item_pointer_t> payload_ranges;

    /// Heap payload length encoded in packets (-1 for unknown)
    s_item_pointer_t heap_length;
    /// Number of bytes of payload received
    s_item_pointer_t received_length;

public:
    /**
     * Freeze a heap. The original heap is destroyed.
     *
     * @param h             The heap to freeze.
     * @param keep_payload  If true, transfer the payload memory allocation from
     *                      the live heap to this object. If false, discard it.
     * @param keep_payload_ranges If true, store information that allows @ref
     *                      get_payload_ranges to work.
     */
    incomplete_heap(live_heap &&h, bool keep_payload, bool keep_payload_ranges);

    /// Heap payload length encoded in packets (-1 for unknown)
    s_item_pointer_t get_heap_length() const { return heap_length; }
    /// Number of bytes of payload received
    s_item_pointer_t get_received_length() const { return received_length; }

    /**
     * Get the payload pointer. This will return an empty pointer unless
     * @a keep_payload was set in the constructor.
     */
    const memory_allocator::pointer &get_payload() const { return payload; }

    /**
     * Return a list of contiguous ranges of payload that were received. This
     * is intended for special cases where a custom memory allocator was used
     * to channel the payload into a caller-managed area, so that the caller
     * knows which parts of that area have been filled in.
     *
     * If @a keep_payload_ranges was @c false in the constructor, returns an
     * empty list.
     */
    std::vector<std::pair<s_item_pointer_t, s_item_pointer_t>> get_payload_ranges() const;
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_HEAP_H
