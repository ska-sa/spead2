/* Copyright 2015 SKA South Africa
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

#include <cassert>
#include <cstring>
#include <algorithm>
#include <utility>
#include "recv_live_heap.h"
#include "recv_utils.h"
#include "common_defines.h"
#include "common_endian.h"
#include "common_logging.h"

namespace spead2
{
namespace recv
{

live_heap::live_heap(s_item_pointer_t cnt, bug_compat_mask bug_compat)
    : cnt(cnt), bug_compat(bug_compat)
{
    assert(cnt >= 0);
}

void live_heap::set_memory_pool(std::shared_ptr<memory_pool> pool)
{
    this->pool = std::move(pool);
}

void live_heap::payload_reserve(std::size_t size, bool exact)
{
    if (size > payload_reserved)
    {
        if (!exact && size < payload_reserved * 2)
        {
            size = payload_reserved * 2;
        }
        memory_pool::pointer new_payload;
        if (pool != nullptr)
            new_payload = pool->allocate(size);
        else
        {
            std::uint8_t *ptr = new std::uint8_t[size];
            new_payload = memory_pool::pointer(ptr, std::default_delete<std::uint8_t[]>());
        }
        if (payload)
            std::memcpy(new_payload.get(), payload.get(), payload_reserved);
        payload = std::move(new_payload);
        payload_reserved = size;
    }
}

bool live_heap::add_packet(const packet_header &packet)
{
    if (cnt != packet.heap_cnt)
    {
        log_debug("packet rejected because HEAP_CNT does not match");
        return false;
    }
    if (heap_length >= 0
        && packet.heap_length >= 0
        && packet.heap_length != heap_length)
    {
        // this could cause overflows later if not caught
        log_debug("packet rejected because its HEAP_LEN is inconsistent with the heap");
        return false;
    }
    if (packet.heap_length >= 0 && packet.heap_length < min_length)
    {
        log_debug("packet rejected because its HEAP_LEN is too small for the heap");
        return false;
    }
    if (heap_address_bits != -1 && packet.heap_address_bits != heap_address_bits)
    {
        log_debug("packet rejected because its flavour is inconsistent with the heap");
        return false;
    }

    // Packet seems sane, check if we've already seen it, and if not, insert it
    bool new_offset = packet_offsets.insert(packet.payload_offset).second;
    if (!new_offset)
    {
        log_debug("packet rejected because it is a duplicate");
        return false;
    }

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
        item_pointer_t pointer = load_be<item_pointer_t>(packet.pointers + i * sizeof(item_pointer_t));
        s_item_pointer_t item_id = decoder.get_id(pointer);
        if (!decoder.is_immediate(pointer))
            min_length = std::max(min_length, s_item_pointer_t(decoder.get_address(pointer)));
        if (item_id == 0 || decoder.get_id(pointer) > PAYLOAD_LENGTH_ID)
        {
            /* NULL items are included because they can be direct-addressed, and this
             * pointer may determine the length of the previous direct-addressed item.
             */
            pointers.push_back(pointer);
            if (item_id == STREAM_CTRL_ID && decoder.is_immediate(pointer)
                && decoder.get_immediate(pointer) == CTRL_STREAM_STOP)
                end_of_stream = true;
        }
    }

    if (packet.payload_length > 0)
    {
        std::memcpy(payload.get() + packet.payload_offset,
                    packet.payload,
                    packet.payload_length);
        received_length += packet.payload_length;
    }
    log_debug("packet with %d bytes of payload at offset %d added to heap %d",
              packet.payload_length, packet.payload_offset, cnt);
    return true;
}

bool live_heap::is_complete() const
{
    return received_length == heap_length;
}

bool live_heap::is_contiguous() const
{
    return received_length == min_length;
}

bool live_heap::is_end_of_stream() const
{
    return end_of_stream;
}

} // namespace recv
} // namespace spead2
