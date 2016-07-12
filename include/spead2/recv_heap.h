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
 * Received heap that has been finalised.
 */
class heap
{
private:
    s_item_pointer_t cnt;       ///< Heap ID
    flavour flavour_;           ///< Flavour
    /**
     * Extracted items. The pointers in the items point into either @ref
     * payload or @ref immediate_payload.
     */
    std::vector<item> items;
    /// Heap payload
    memory_allocator::pointer payload;
    /// Storage for immediate values
    std::unique_ptr<std::uint8_t[]> immediate_payload;

public:
    /**
     * Freeze a heap, which must satisfy live_heap::is_contiguous. The original
     * heap is destroyed.
     */
    explicit heap(live_heap &&h);
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

    /**
     * Convenience function to check whether any of the items is
     * a @c CTRL_STREAM_START.
     */
    bool is_start_of_stream() const;
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_HEAP_H
