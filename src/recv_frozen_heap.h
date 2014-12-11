/**
 * @file
 */

#ifndef SPEAD_RECV_FROZEN_HEAP
#define SPEAD_RECV_FROZEN_HEAP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "common_defines.h"

class heap;

namespace spead
{
namespace recv
{

/**
 * An item extracted from a heap.
 *
 * @todo Move this up to the spead namespace?
 */
struct item
{
    /// Item ID
    std::int64_t id;
    /// Item mode (immediate or addressed)
    bool is_immediate;
    union
    {
        std::int64_t immediate; ///< Immediate value (if @ref is_immediate)
        /// Pointer and length (if @ref is_immediate is false)
        struct
        {
            std::uint8_t *ptr;
            std::size_t length;
        } address;
    } value;
};

/**
 * Received heap that has been finalised.
 */
class frozen_heap
{
private:
    std::int64_t heap_cnt;    ///< Heap ID
    int heap_address_bits;    ///< Flavour
    /**
     * Extracted items. The pointers in the items point to @ref payload.
     */
    std::vector<item> items;
    /// Heap payload
    std::unique_ptr<std::uint8_t[]> payload;

public:
    /**
     * Freeze a heap, which must satisfy heap::is_contiguous. The original
     * heap is destroyed.
     */
    frozen_heap(heap &&h);
    /// Get heap ID
    std::int64_t cnt() const { return heap_cnt; }
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
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_FROZEN_HEAP
