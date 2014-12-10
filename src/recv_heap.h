#ifndef SPEAD_RECV_HEAP
#define SPEAD_RECV_HEAP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_set>
#include "recv_packet.h"

namespace spead
{
namespace recv
{

class frozen_heap;

class heap
{
private:
    friend class frozen_heap;

    std::int64_t heap_cnt = -1;
    std::int64_t heap_length = -1;
    std::int64_t received_length = 0;
    std::int64_t min_length = 0;      // length implied by packet payloads
    int heap_address_bits = -1;
    // We don't use std::vector because it zero-fills
    std::unique_ptr<uint8_t[]> payload;
    std::size_t payload_reserved = 0;
    std::vector<std::uint64_t> pointers;
    /* TODO: investigate more efficient structures here, e.g.
     * - a bitfield (one bit per payload byte)
     * - using a Bloom filter first
     * - using a linked list per offset>>13 (which is maybe equivalent to
     *   just changing the hash function)
     * This is used only to detect duplicate packets, so fits the use case
     * for a Bloom filter nicely.
     */
    std::unordered_set<std::int64_t> packet_offsets;

    /**
     * Make sure at least @a size bytes are allocated for payload. If
     * @a exact is false, then a doubling heuristic will be used.
     */
    void payload_reserve(std::size_t size, bool exact);

public:
    heap() = default;
    explicit heap(std::int64_t heap_cnt);
    bool add_packet(const packet_header &packet);
    // True if we have received a heap size header and all of it has been received
    bool is_complete() const;
    // True if all the payload we have received all data up to min_length
    bool is_contiguous() const;
    std::int64_t cnt() const { return heap_cnt; }
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_HEAP
