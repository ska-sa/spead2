/**
 * @file
 */

#include <cassert>
#include <cstring>
#include <algorithm>
#include "recv_heap.h"
#include "recv_utils.h"
#include "common_defines.h"

namespace spead
{
namespace recv
{

heap::heap(std::int64_t heap_cnt) : heap_cnt(heap_cnt)
{
    assert(heap_cnt >= 0);
}

void heap::payload_reserve(std::size_t size, bool exact)
{
    if (size > payload_reserved)
    {
        if (!exact && size < payload_reserved * 2)
        {
            size = payload_reserved * 2;
        }
        std::unique_ptr<std::uint8_t[]> new_payload(new std::uint8_t[size]);
        if (payload)
            std::memcpy(new_payload.get(), payload.get(), payload_reserved);
        payload = std::move(new_payload);
        payload_reserved = size;
    }
}

bool heap::add_packet(const packet_header &packet)
{
    if (heap_cnt != packet.heap_cnt)
        return false;     // wrong heap ID
    if (heap_length >= 0
        && packet.heap_length >= 0
        && packet.heap_length != heap_length)
        return false;     // inconsistent heap lengths - could cause trouble later
    if (packet.heap_length >= 0 && packet.heap_length < min_length)
        return false;     // inconsistent with already-seen payloads
    if (heap_address_bits != -1 && packet.heap_address_bits != heap_address_bits)
        return false;     // flavour mismatch

    // Packet seems sane, check if we've already seen it, and if not, insert it
    bool new_offset = packet_offsets.insert(packet.payload_offset).second;
    if (!new_offset)
        return false;

    ///////////////////////////////////////////////
    // Packet is now accepted, and we modify state
    ///////////////////////////////////////////////

    heap_address_bits = packet.heap_address_bits;
    // If this is the first time we know the length, record it
    if (heap_length < 0 && packet.heap_length >= 0)
    {
        heap_length = packet.heap_length;
        min_length = heap_length;
        payload_reserve(heap_length, true);
    }
    min_length = std::max(min_length, packet.payload_offset + packet.payload_length);
    pointer_decoder decoder(heap_address_bits);
    for (int i = 0; i < packet.n_items; i++)
    {
        // TODO: handle stream control here
        std::uint64_t pointer = be64toh(packet.pointers[i]);
        std::int64_t item_id = decoder.get_id(pointer);
        if (!decoder.is_immediate(pointer))
            min_length = std::max(min_length, std::int64_t(decoder.get_address(pointer)));
        if (item_id == 0 || decoder.get_id(pointer) > PAYLOAD_LENGTH_ID)
        {
            /* NULL items are included because they can be direct-addressed, and this
             * pointer may determine the length of the previous direct-addressed item.
             */
            pointers.push_back(pointer);
        }
    }

    if (packet.payload_length > 0)
    {
        std::memcpy(payload.get() + packet.payload_offset,
                    packet.payload,
                    packet.payload_length);
        received_length += packet.payload_length;
    }
    return true;
}

bool heap::is_complete() const
{
    return received_length == heap_length;
}

bool heap::is_contiguous() const
{
    return received_length == min_length;
}

} // namespace recv
} // namespace spead
