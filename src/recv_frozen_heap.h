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

struct item
{
    std::int64_t id;
    bool is_immediate;
    union
    {
        std::int64_t immediate; // unused if data != NULL
        struct
        {
            const std::uint8_t *ptr;
            std::size_t length;
        } address;
    } value;
};

class frozen_heap
{
private:
    std::int64_t heap_cnt;
    int heap_address_bits;
    std::vector<item> items;
    std::unique_ptr<std::uint8_t[]> payload;

public:
    /**
     * Freeze a heap, which must satisfy heap::is_contiguous. The original
     * heap is destroyed.
     */
    frozen_heap(heap &&h);
    std::int64_t cnt() const { return heap_cnt; }
    const std::vector<item> &get_items() const { return items; }

    // Extract descriptor fields from the heap. Any missing fields are
    // default-initialized. This should be used on a heap constructed from
    // the content of a descriptor item.
    descriptor to_descriptor() const;

    // Extract and decode descriptors from this heap
    std::vector<descriptor> get_descriptors() const;

    // Extract descriptors from this heap and update metadata
    void update_descriptors(descriptor_map &descriptors) const;
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_FROZEN_HEAP
