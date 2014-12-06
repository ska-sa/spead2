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
    std::uint16_t heap_address_bits;
    std::uint16_t n_items;
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
 * Currently, @a size must match the packet contents to be considered value.
 * TODO: come up with a mechanism to let the size be detected?
 *
 * @pre @a raw is 8-byte aligned
 */
std::size_t decode_packet(packet_header &out, const uint8_t *raw, std::size_t max_size);

class heap
{
private:
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
    bool is_complete() const;
    std::int64_t cnt() const { return heap_cnt; }
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
};

} // namespace in
} // namespace spead

#endif // SPEAD_IN_H
