/**
 * @file
 */

#ifndef SPEAD_SEND_PACKET_H
#define SPEAD_SEND_PACKET_H

#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <boost/asio/buffer.hpp>

namespace spead
{
namespace send
{

class basic_heap;

/**
 * A packet ready for sending on the network. It contains some internally
 * held data, and const buffer sequence that contains a mix of pointers to
 * the internal data and pointers to the heap's items.
 *
 * If @a buffers is empty, it indicates the end of the heap.
 *
 * @todo Investigate whether number of new calls could be reduced by using
 * a pool for the case of packets with no item pointers other than the
 * per-packet ones.
 */
struct packet
{
    std::unique_ptr<std::uint8_t[]> data;
    std::vector<boost::asio::const_buffer> buffers;
};

class packet_generator
{
private:
    // 8 bytes head, 8 bytes each for heap cnt, heap size, payload offset, payload size
    static constexpr std::size_t prefix_size = 40;

    const basic_heap &h;
    int heap_address_bits;
    std::size_t max_packet_size;
    std::size_t max_item_pointers_per_packet;

    /// Next item pointer to send
    std::size_t next_item_pointer = 0;
    /// Current item payload being sent
    std::size_t next_item = 0;
    /// Amount of next_item already sent
    std::size_t next_item_offset = 0;
    /// Address at which payload for the next item will be found
    std::size_t next_address = 0;
    /// Payload offset for the next packet
    std::int64_t payload_offset = 0;
    std::int64_t payload_size = 0;

public:
    packet_generator(const basic_heap &h, int heap_address_bits, std::size_t max_packet_size);

    packet next_packet();
};

} // namespace send
} // namespace spead

#endif // SPEAD_SEND_PACKET_H
