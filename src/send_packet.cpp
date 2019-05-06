/* Copyright 2015, 2019 SKA South Africa
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

#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <spead2/send_heap.h>
#include <spead2/send_utils.h>
#include <spead2/send_packet.h>
#include <spead2/common_defines.h>
#include <spead2/common_logging.h>
#include <spead2/common_endian.h>

namespace spead2
{
namespace send
{

constexpr std::size_t packet_generator::prefix_size;

static bool use_immediate(const item &it, std::size_t max_immediate_size)
{
    return it.is_inline
        || (it.allow_immediate && it.data.buffer.length <= max_immediate_size);
}

packet_generator::packet_generator(
    const heap &h, item_pointer_t cnt, std::size_t max_packet_size)
    : h(h), cnt(cnt), max_packet_size(max_packet_size)
{
    // Round down max packet size so that we can align payload
    max_packet_size &= ~7;
    /* We need
     * - the prefix
     * - an item pointer
     * - sizeof(item_pointer) bytes of payload, to ensure unique payload offsets
     * (actually 1 byte is enough, but it is better to keep payload aligned)
     */
    if (max_packet_size < prefix_size + 2 * sizeof(item_pointer_t))
        throw std::invalid_argument("packet size is too small");

    payload_size = 0;
    const std::size_t max_immediate_size = h.get_flavour().get_heap_address_bits() / 8;
    for (const item &it : h.items)
    {
        if (!use_immediate(it, max_immediate_size))
            payload_size += it.data.buffer.length;
    }

    /* Check if we need to add dummy payload to ensure that every packet
     * contains some payload.
     */
    max_item_pointers_per_packet =
        (max_packet_size - (prefix_size + sizeof(item_pointer_t))) / sizeof(item_pointer_t);
    /* Number of packets needed to send all the item pointers, plus
     * potentially one extra in case we need to inject a NULL item pointer
     * to mark the separation between the padding and the last item.
     */
    std::size_t item_packets = h.items.size() / max_item_pointers_per_packet + 1;
    if (h.get_repeat_pointers() && item_packets > 1)
        throw std::invalid_argument("packet size is too small to repeat item pointers");
    /* We want every packet to have some payload, so that packets can be
     * unambiguously ordered and lost packets can be detected. For all
     * packets except the last, we want a multiple of sizeof(item_pointer_t)
     * bytes, so that copies are nicely aligned.
     */
    std::int64_t min_payload_size = s_item_pointer_t(item_packets - 1) * sizeof(item_pointer_t) + 1;
    if (payload_size < min_payload_size)
    {
        log_debug("Increase payload size from %d to %d", payload_size, min_payload_size);
        payload_size = min_payload_size;
        need_null_item = true;
    }
}

bool packet_generator::has_next_packet() const
{
    return payload_offset < payload_size;
}

packet packet_generator::next_packet()
{
    packet out;

    if (h.get_repeat_pointers())
    {
        next_item_pointer = 0;
        next_address = 0;
    }

    if (payload_offset < payload_size)
    {
        pointer_encoder encoder(h.get_flavour().get_heap_address_bits());
        const std::size_t max_immediate_size = h.get_flavour().get_heap_address_bits() / 8;
        const std::size_t n_item_pointers = std::min(
            max_item_pointers_per_packet,
            h.items.size() + need_null_item - next_item_pointer);
        std::size_t packet_payload_length = std::min(
            std::size_t(payload_size - payload_offset),
            max_packet_size - n_item_pointers * sizeof(item_pointer_t) - prefix_size);

        // Determine how much internal data is needed.
        // Always add enough to allow for padding the payload
        std::size_t alloc_bytes = prefix_size + (n_item_pointers + 1) * sizeof(item_pointer_t);
        out.data.reset(new std::uint8_t[alloc_bytes]);
        std::uint64_t *header = reinterpret_cast<std::uint64_t *>(out.data.get());
        *header = htobe<std::uint64_t>(
            (std::uint64_t(0x5304) << 48)
            | (std::uint64_t(8 - max_immediate_size) << 40)
            | (std::uint64_t(max_immediate_size) << 32)
            | (n_item_pointers + 4));
        // TODO: if item_pointer_t is more than 64 bits, this will misalign
        item_pointer_t *pointer = reinterpret_cast<item_pointer_t *>(out.data.get() + 8);
        *pointer++ = htobe<item_pointer_t>(encoder.encode_immediate(HEAP_CNT_ID, cnt));
        *pointer++ = htobe<item_pointer_t>(encoder.encode_immediate(HEAP_LENGTH_ID, payload_size));
        *pointer++ = htobe<item_pointer_t>(encoder.encode_immediate(PAYLOAD_OFFSET_ID, payload_offset));
        *pointer++ = htobe<item_pointer_t>(encoder.encode_immediate(PAYLOAD_LENGTH_ID, packet_payload_length));
        for (std::size_t i = 0; i < n_item_pointers; i++)
        {
            item_pointer_t ip;
            if (next_item_pointer == h.items.size())
            {
                assert(need_null_item);
                ip = htobe<item_pointer_t>(encoder.encode_address(NULL_ID, next_address));
            }
            else
            {
                const item &it = h.items[next_item_pointer];
                if (it.is_inline)
                {
                    ip = htobe<item_pointer_t>(encoder.encode_immediate(it.id, it.data.immediate));
                }
                else if (it.allow_immediate && it.data.buffer.length <= max_immediate_size)
                {
                    ip = htobe<item_pointer_t>(encoder.encode_immediate(it.id, 0));
                    std::memcpy(reinterpret_cast<char *>(&ip) + sizeof(item_pointer_t) - it.data.buffer.length,
                                it.data.buffer.ptr, it.data.buffer.length);
                }
                else
                {
                    ip = htobe<item_pointer_t>(encoder.encode_address(it.id, next_address));
                    next_address += it.data.buffer.length;
                }
            }
            *pointer++ = ip;
            next_item_pointer++;
        }
        out.buffers.emplace_back(out.data.get(), prefix_size + 8 * n_item_pointers);

        // Generate payload
        payload_offset += packet_payload_length;
        while (packet_payload_length > 0)
        {
            if (next_item == h.items.size())
            {
                // Dummy padding payload. Fill with zeros to simplify testing
                assert(need_null_item);
                assert(packet_payload_length <= 8);
                *pointer = 0;
                out.buffers.emplace_back(pointer, packet_payload_length);
                packet_payload_length = 0;
            }
            else if (use_immediate(h.items[next_item], max_immediate_size))
            {
                next_item++;
                next_item_offset = 0;
            }
            else
            {
                const item &it = h.items[next_item];
                std::size_t send_bytes = std::min(
                    it.data.buffer.length - next_item_offset, packet_payload_length);
                out.buffers.emplace_back(it.data.buffer.ptr + next_item_offset, send_bytes);
                next_item_offset += send_bytes;
                if (next_item_offset == it.data.buffer.length)
                {
                    next_item++;
                    next_item_offset = 0;
                }
                packet_payload_length -= send_bytes;
            }
        }
    }
    return out;
}

} // namespace send
} // namespace spead2
