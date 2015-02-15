/**
 * @file
 */

#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <endian.h>
#include "send_heap.h"
#include "send_utils.h"
#include "common_defines.h"

namespace spead
{
namespace send
{

constexpr std::size_t packet_generator::prefix_size;

packet_generator::packet_generator(
    const heap &h, int heap_address_bits, std::size_t max_packet_size)
    : h(h), heap_address_bits(heap_address_bits), max_packet_size(max_packet_size)
{
    /* We need
     * - the prefix
     * - 8 bytes to send an item pointer
     * - 8 bytes of payload, to ensure unique payload offsets
     * (actually 1 byte is enough, but it is better to keep payload aligned)
     */
    if (max_packet_size < prefix_size + 16)
        throw std::invalid_argument("packet size is too small");

    payload_size = 0;
    const std::size_t max_immediate_size = heap_address_bits / 8;
    for (const item &it : h.items)
    {
        if (it.length > max_immediate_size)
            payload_size += it.length;
    }

    /* Check if we need to add dummy payload to ensure that every packet
     * contains some payload.
     */
    max_item_pointers_per_packet = (max_packet_size - (prefix_size + 8)) / 8;
    // Number of packets needed to send all the item pointers
    std::size_t item_packets = (h.items.size() + max_item_pointers_per_packet - 1) / max_item_pointers_per_packet;
    payload_size = std::max(payload_size, std::int64_t(item_packets) * 8);
}

packet packet_generator::next_packet()
{
    packet out;

    if (payload_offset < payload_size)
    {
        pointer_encoder encoder(heap_address_bits);
        std::size_t max_immediate_size = heap_address_bits / 8;
        std::size_t immediate_offset = 8 - max_immediate_size;
        std::size_t n_item_pointers = std::min(max_item_pointers_per_packet, next_item_pointer - h.items.size());
        std::size_t packet_payload_length = std::min(
            payload_size - payload_offset,
            std::int64_t(max_packet_size - n_item_pointers * 8 - prefix_size));

        // Determine how much internal data is needed.
        // Always add enough to allow for padding the payload
        std::size_t alloc_bytes = prefix_size + n_item_pointers + 8;
        out.data.reset(new std::uint8_t[alloc_bytes]);
        std::uint64_t *header = reinterpret_cast<std::uint64_t *>(out.data.get());
        *header++ = htobe64(
            (std::uint64_t(0x5304) << 48)
            | (std::uint64_t(8 - heap_address_bits / 8) << 40)
            | (std::uint64_t(heap_address_bits / 8) << 32)
            | (h.items.size()));
        *header++ = htobe64(encoder.encode_immediate(HEAP_CNT_ID, h.heap_cnt));
        *header++ = htobe64(encoder.encode_immediate(HEAP_LENGTH_ID, payload_size));
        *header++ = htobe64(encoder.encode_immediate(PAYLOAD_OFFSET_ID, payload_offset));
        *header++ = htobe64(encoder.encode_immediate(PAYLOAD_LENGTH_ID, packet_payload_length));
        for (std::size_t i = 0; i < n_item_pointers; i++)
        {
            const item &it = h.items[next_item_pointer++];
            std::uint64_t ip;
            if (it.length <= max_immediate_size)
            {
                ip = htobe64(encoder.encode_immediate(it.id, 0));
                std::memcpy(reinterpret_cast<char *>(&ip) + immediate_offset, it.ptr, it.length);
            }
            else
            {
                ip = htobe64(encoder.encode_address(it.id, next_address));
                next_address += it.length;
            }
            *header++ = ip;
        }
        out.buffers.emplace_back(out.data.get(), prefix_size + 8 * n_item_pointers);

        // Generate payload
        while (packet_payload_length > 0)
        {
            if (next_item == h.items.size())
            {
                // dummy padding payload
                out.buffers.emplace_back(header, packet_payload_length);
            }
            else if (h.items[next_item].length <= max_immediate_size)
            {
                next_item++;
                next_item_offset = 0;
            }
            else
            {
                const item &it = h.items[next_item];
                std::size_t send_bytes = std::min(it.length - next_item_offset, packet_payload_length);
                out.buffers.emplace_back(it.ptr + next_item_offset, send_bytes);
                next_item_offset += send_bytes;
                if (next_item_offset == it.length)
                {
                    next_item++;
                    next_item_offset = 0;
                }
                packet_payload_length -= send_bytes;
            }
        }
        payload_offset += packet_payload_length;
    }
    return out;
}

} // namespace send
} // namespace spead
