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
#include <stdexcept>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_utils.h>
#include <spead2/common_defines.h>
#include <spead2/common_endian.h>
#include <spead2/common_logging.h>

namespace spead2
{
namespace recv
{

live_heap::live_heap(s_item_pointer_t cnt, bug_compat_mask bug_compat,
                     std::shared_ptr<memory_allocator> allocator)
    : cnt(cnt), bug_compat(bug_compat), allocator(std::move(allocator))
{
    assert(this->allocator);
    assert(cnt >= 0);
}

void live_heap::set_memcpy(memcpy_function memcpy)
{
    this->memcpy = memcpy;
}

void live_heap::payload_reserve(std::size_t size, bool exact, const packet_header &packet)
{
    if (size > payload_reserved)
    {
        if (!exact && size < payload_reserved * 2)
        {
            size = payload_reserved * 2;
        }
        memory_allocator::pointer new_payload;
        new_payload = allocator->allocate(size, (void *) &packet);
        if (payload)
            this->memcpy(new_payload.get(), payload.get(), payload_reserved);
        payload = std::move(new_payload);
        payload_reserved = size;
    }
}

bool live_heap::add_payload_range(s_item_pointer_t first, s_item_pointer_t last)
{
    decltype(payload_ranges)::iterator prev, next, ptr;
    next = payload_ranges.upper_bound(first);
    if (next != payload_ranges.end() && next->first < last)
    {
        log_warning("packet rejected because it partially overlaps existing payload");
        return false;
    }
    else if (next == payload_ranges.begin()
        || (prev = std::prev(next))->second < first)
    {
        // The prior range, if any, does not intersect this one
        ptr = payload_ranges.emplace_hint(next, first, last);
    }
    else if (prev->second == first)
    {
        prev->second = last;
        ptr = prev;
    }
    else
    {
        /* Only debug level for this, because it can legitimately happen in the
         * network. There are also ways this can happen with a partial overlap
         * instead of a duplicate, but it would cost more cycles to test for
         * it.
         */
        log_debug("packet rejected because it is a duplicate");
        return false;
    }

    if (next != payload_ranges.end() && next->first == last)
    {
        ptr->second = next->second;
        payload_ranges.erase(next);
    }
    return true;
}

bool live_heap::add_packet(const packet_header &packet)
{
    assert(cnt == packet.heap_cnt);
    if (heap_length >= 0
        && packet.heap_length >= 0
        && packet.heap_length != heap_length)
    {
        // this could cause overflows later if not caught
        log_info("packet rejected because its HEAP_LEN is inconsistent with the heap");
        return false;
    }
    if (packet.heap_length >= 0 && packet.heap_length < min_length)
    {
        log_info("packet rejected because its HEAP_LEN is too small for the heap");
        return false;
    }
    if (heap_address_bits != -1 && packet.heap_address_bits != heap_address_bits)
    {
        log_info("packet rejected because its flavour is inconsistent with the heap");
        return false;
    }

    // Packet seems sane, check if we've already seen it, and if not, insert it
    bool new_packet = add_payload_range(packet.payload_offset,
                                        packet.payload_offset + packet.payload_length);
    if (!new_packet)
        return false;

    ///////////////////////////////////////////////
    // Packet is now accepted, and we modify state
    ///////////////////////////////////////////////

    heap_address_bits = packet.heap_address_bits;
    // If this is the first time we know the length, record it
    if (heap_length < 0 && packet.heap_length >= 0)
    {
        heap_length = packet.heap_length;
        min_length = std::max(min_length, heap_length);
        payload_reserve(min_length, true, packet);
    }
    else
    {
        min_length = std::max(min_length, packet.payload_offset + packet.payload_length);
        payload_reserve(min_length, false, packet);
    }
    pointer_decoder decoder(heap_address_bits);
    /* Try to avoid too many reallocations for pointers. We can't just
     * unconditionally call reserve for the size we actually want, because we
     * should avoid disabling vector's doubling heuristic.
     */
    if (pointers.capacity() == 0)
        pointers.reserve(packet.n_items);
    for (int i = 0; i < packet.n_items; i++)
    {
        item_pointer_t pointer = load_be<item_pointer_t>(packet.pointers + i * sizeof(item_pointer_t));
        if (seen_pointers.insert(pointer).second) // true if this is a new addition
        {
            if (!decoder.is_immediate(pointer))
                min_length = std::max(min_length, s_item_pointer_t(decoder.get_address(pointer)));
            s_item_pointer_t item_id = decoder.get_id(pointer);
            if (item_id == 0 || item_id > PAYLOAD_LENGTH_ID)
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
    }

    if (packet.payload_length > 0)
    {
        this->memcpy(payload.get() + packet.payload_offset,
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
    /* The check against min_length is purely a sanity check on the packet:
     * if it contains item pointer offsets that are bigger than the explicitly
     * provided heap length, it cannot be decoded correctly.
     */
    return received_length == heap_length && received_length == min_length;
}

bool live_heap::is_contiguous() const
{
    return received_length == min_length;
}

bool live_heap::is_end_of_stream() const
{
    return end_of_stream;
}

s_item_pointer_t live_heap::get_received_length() const
{
    return received_length;
}

s_item_pointer_t live_heap::get_heap_length() const
{
    return heap_length;
}

} // namespace recv
} // namespace spead2
