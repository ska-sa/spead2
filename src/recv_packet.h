#ifndef SPEAD_RECV_PACKET
#define SPEAD_RECV_PACKET

#include <cstddef>
#include <cstdint>

namespace spead
{
namespace recv
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

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_PACKET
