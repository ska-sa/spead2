#ifndef SPEAD_IN_H
#define SPEAD_IN_H

#include <cstdint>
#include <vector>
#include <deque>
#include <unordered_set>
#include <memory>
#include <functional>
#include "defines.h"

namespace spead
{

namespace in
{

struct packet_header
{
    int heap_address_bits;
    int n_items;
    // Real values for the below are non-negative, but -1 indicates missing
    std::int64_t heap_cnt;
    std::int64_t heap_length;
    std::int64_t payload_offset;
    std::int64_t payload_length;
    const uint64_t *pointers;   // in big endian
    const uint8_t *payload;
};

/**
 * Split out the header fields for the packet. On success, returns the
 * length of the packet. On failure, returns 0, and @a out is undefined.
 * TODO: use system error code mechanisms to report failures?
 *
 * @pre @a raw is 8-byte aligned
 */
std::size_t decode_packet(packet_header &out, const uint8_t *raw, std::size_t max_size);

class heap
{
private:
    friend class frozen_heap;

    std::int64_t heap_cnt;
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
    explicit heap(std::int64_t heap_cnt);
    bool add_packet(const packet_header &packet);
    // True if we have received a heap size header and all of it has been received
    bool is_complete() const;
    // True if all the payload we have received all data up to min_length
    bool is_contiguous() const;
    std::int64_t cnt() const { return heap_cnt; }
};

class frozen_heap
{
public:
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

class stream
{
private:
    // TODO: replace with a fixed-size ring buffer
    std::size_t max_heaps;
    std::deque<heap> heaps;
    std::function<void(heap &&)> callback;

public:
    explicit stream(std::size_t max_heaps = 16);
    void set_callback(std::function<void(heap &&)> callback);
    void set_max_heaps(std::size_t max_heaps);
    bool add_packet(const uint8_t *data, std::size_t size);

    // Clear out all heaps from the deque, even if not complete
    void flush();
};

} // namespace in
} // namespace spead

#endif // SPEAD_IN_H
