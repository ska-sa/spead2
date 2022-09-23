/* Copyright 2021-2022 National Research Foundation (SARAO)
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
#include <memory>
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <utility>
#include <spead2/common_defines.h>
#include <spead2/common_endian.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/recv_packet.h>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_heap.h>
#include <spead2/recv_utils.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_chunk_stream.h>

namespace spead2
{
namespace recv
{

constexpr std::size_t chunk_stream_config::default_max_chunks;

chunk_stream_config &chunk_stream_config::set_items(const std::vector<item_pointer_t> &item_ids)
{
    this->item_ids = item_ids;
    return *this;
}

chunk_stream_config &chunk_stream_config::set_max_chunks(std::size_t max_chunks)
{
    if (max_chunks == 0)
        throw std::invalid_argument("max_chunks cannot be 0");
    this->max_chunks = max_chunks;
    return *this;
}

chunk_stream_config &chunk_stream_config::set_place(chunk_place_function place)
{
    this->place = std::move(place);
    return *this;
}

chunk_stream_config &chunk_stream_config::set_allocate(chunk_allocate_function allocate)
{
    this->allocate = std::move(allocate);
    return *this;
}

chunk_stream_config &chunk_stream_config::set_ready(chunk_ready_function ready)
{
    this->ready = std::move(ready);
    return *this;
}

chunk_stream_config &chunk_stream_config::enable_packet_presence(std::size_t payload_size)
{
    if (payload_size == 0)
        throw std::invalid_argument("payload_size must not be zero");
    this->packet_presence_payload_size = payload_size;
    return *this;
}

chunk_stream_config &chunk_stream_config::disable_packet_presence()
{
    packet_presence_payload_size = 0;
    return *this;
}

chunk_stream_config &chunk_stream_config::set_max_heap_extra(std::size_t max_heap_extra)
{
    this->max_heap_extra = max_heap_extra;
    return *this;
}


namespace detail
{

// Round a size up to the next multiple of align
static std::size_t round_up(std::size_t size, std::size_t align)
{
    return (size + align - 1) / align * align;
}

chunk_stream_state::chunk_stream_state(
    const stream_config &config, const chunk_stream_config &chunk_config)
    : orig_memcpy(config.get_memcpy()),
    chunk_config(chunk_config),
    stream_id(config.get_stream_id()),
    base_stat_index(config.next_stat_index()),
    chunks(chunk_config.get_max_chunks())
{
    if (!this->chunk_config.get_place())
        throw std::invalid_argument("chunk_config.place is not set");
    if (!this->chunk_config.get_allocate())
        throw std::invalid_argument("chunk_config.allocate is not set");
    if (!this->chunk_config.get_ready())
        throw std::invalid_argument("chunk_config.ready is not set");

    /* Compute the memory required for place_data_storage. The layout is
     * - chunk_place_data
     * - item pointers (with s_item_pointer_t alignment)
     * - extra (with max_align_t alignment)
     */
    constexpr std::size_t max_align = alignof(std::max_align_t);
    const std::size_t n_items = chunk_config.get_items().size();
    std::size_t space = sizeof(chunk_place_data);
    space = round_up(space, alignof(s_item_pointer_t));
    std::size_t item_offset = space;
    space += n_items * sizeof(s_item_pointer_t);
    space = round_up(space, max_align);
    std::size_t extra_offset = space;
    space += chunk_config.get_max_heap_extra();
    /* operator new is required to return a pointer suitably aligned for an
     * object of the requested size. Round up to a multiple of max_align so
     * that the library cannot infer a smaller alignment.
     */
    space = round_up(space, max_align);

    /* Allocate the memory, and use placement new to initialise it. For the
     * arrays the placement new shouldn't actually run any code, but it
     * officially starts the lifetime of the object in terms of the C++ spec.
     * It's not if it's actually portable in C++11: implementations used to be
     * allowed to add overhead for array new, even when using placement new.
     * CWG 2382 disallowed that for placement new, and in practice it sounds
     * like no compiler ever added overhead for scalar types (MSVC used to do
     * it for polymorphic classes).
     *
     * In C++20 it's probably not necessary to use the placement new due to
     * the rules about implicit-lifetime types, although the examples imply
     * it is necessary to use std::launder so it wouldn't be any simpler.
     */
    unsigned char *ptr = reinterpret_cast<unsigned char *>(operator new(space));
    place_data = new(ptr) chunk_place_data();
    if (n_items > 0)
        place_data->items = new(ptr + item_offset) s_item_pointer_t[n_items];
    else
        place_data->items = nullptr;
    if (chunk_config.get_max_heap_extra() > 0)
        place_data->extra = new(ptr + extra_offset) std::uint8_t[chunk_config.get_max_heap_extra()];
    else
        place_data->extra = nullptr;
    place_data_storage.reset(ptr);
}

void chunk_stream_state::free_place_data::operator()(unsigned char *ptr)
{
    // TODO: should this use std::launder in C++17?
    auto *place_data = reinterpret_cast<chunk_place_data *>(ptr);
    place_data->~chunk_place_data();
    operator delete[](ptr);
}

void chunk_stream_state::packet_memcpy(
    const memory_allocator::pointer &allocation,
    const packet_header &packet) const
{
    const heap_metadata &metadata = *get_heap_metadata(allocation);
    if (metadata.chunk_id < head_chunk)
    {
        // The packet corresponds to a chunk that has already been aged out
        // TODO: increment a counter / log a warning
        return;
    }
    orig_memcpy(allocation, packet);
    std::size_t payload_divide = chunk_config.get_packet_presence_payload_size();
    if (payload_divide != 0)
    {
        // TODO: could possibly optimise this using something like libdivide
        std::size_t index = metadata.heap_index + packet.payload_offset / payload_divide;
        assert(index < metadata.chunk_ptr->present_size);
        metadata.chunk_ptr->present[index] = true;
    }
}

stream_config chunk_stream_state::adjust_config(const stream_config &config)
{
    using namespace std::placeholders;
    stream_config new_config = config;
    // Unsized heaps won't work with the custom allocator
    new_config.set_allow_unsized_heaps(false);
    new_config.set_memory_allocator(std::make_shared<chunk_stream_allocator>(*this));
    // Override the original memcpy with our custom version
    new_config.set_memcpy(std::bind(&chunk_stream_state::packet_memcpy, this, _1, _2));
    // Add custom statistics
    new_config.add_stat("too_old_heaps");
    new_config.add_stat("rejected_heaps");
    return new_config;
}

void chunk_stream_state::flush_head()
{
    assert(head_chunk < tail_chunk);
    if (chunks[head_pos])
    {
        std::uint64_t *batch_stats = static_cast<chunk_stream *>(this)->batch_stats.data();
        chunk_config.get_ready()(std::move(chunks[head_pos]), batch_stats);
        // If the ready callback didn't take over ownership, free it.
        chunks[head_pos].reset();
    }
    head_chunk++;
    head_pos++;
    if (head_pos == chunks.size())
        head_pos = 0;  // wrap around the circular buffer
}

void chunk_stream_state::flush_chunks()
{
    while (head_chunk != tail_chunk)
        flush_head();
}

const chunk_stream_state::heap_metadata *chunk_stream_state::get_heap_metadata(
    const memory_allocator::pointer &ptr)
{
    return ptr.get_deleter().target<heap_metadata>();
}

// Used to get a non-null pointer
static std::uint8_t dummy_uint8;

// Keep these in sync with stats added in adjust_config
static constexpr std::size_t too_old_heaps_offset = 0;
static constexpr std::size_t rejected_heaps_offset = 1;

std::pair<std::uint8_t *, chunk_stream_state::heap_metadata>
chunk_stream_state::allocate(std::size_t size, const packet_header &packet)
{
    /* Extract the user's requested items.
     * TODO: this could possibly be optimised with a hash table (with a
     * perfect hash function chosen in advance), but for the expected
     * sizes the overheads will probably outweight the benefits.
     */
    const auto &item_ids = get_chunk_config().get_items();
    std::fill(place_data->items, place_data->items + item_ids.size(), -1);
    pointer_decoder decoder(packet.heap_address_bits);
    /* packet.pointers and packet.n_items skips initial "special" item
     * pointers. To allow them to be matched as well, we start from the
     * original packet and skip over the 8-byte header.
     */
    for (const std::uint8_t *p = packet.packet + 8; p != packet.payload; p += sizeof(item_pointer_t))
    {
        item_pointer_t pointer = load_be<item_pointer_t>(p);
        if (decoder.is_immediate(pointer))
        {
            item_pointer_t id = decoder.get_id(pointer);
            for (std::size_t j = 0; j < item_ids.size(); j++)
                if (item_ids[j] == id)
                    place_data->items[j] = decoder.get_immediate(pointer);
        }
    }

    /* TODO: see if the storage can be in the class with the deleter
     * just referencing it. That will avoid the implied memory allocation
     * in constructing the std::function underlying the deleter.
     */
    std::pair<std::uint8_t *, heap_metadata> out;
    out.first = &dummy_uint8;  // Use a non-null value to avoid confusion with empty pointers
    heap_metadata &metadata = out.second;

    place_data->packet = packet.packet;
    place_data->packet_size = packet.payload + packet.payload_length - packet.packet;
    place_data->chunk_id = -1;
    place_data->heap_index = 0;
    place_data->heap_offset = 0;
    place_data->batch_stats = static_cast<chunk_stream *>(this)->batch_stats.data();
    place_data->extra_offset = 0;
    place_data->extra_size = 0;
    chunk_config.get_place()(place_data, sizeof(*place_data));
    auto chunk_id = place_data->chunk_id;
    if (chunk_id < head_chunk)
    {
        // We don't want this heap.
        metadata.chunk_id = -1;
        metadata.chunk_ptr = nullptr;
        std::size_t stat_offset = (chunk_id >= 0) ? too_old_heaps_offset : rejected_heaps_offset;
        place_data->batch_stats[base_stat_index + stat_offset]++;
        return out;
    }
    else
    {
        std::size_t max_chunks = chunk_config.get_max_chunks();
        if (chunk_id >= tail_chunk)
        {
            // We've moved beyond the end of our current window, and need to
            // allocate fresh chunks.
            const auto &allocate = chunk_config.get_allocate();
            if (chunk_id >= tail_chunk + std::int64_t(max_chunks))
            {
                /* We've jumped ahead so far that the entire current window
                 * is stale. Flush it all and fast-forward to the new window.
                 * We leave it to the while loop below to actually allocate
                 * the chunks.
                 */
                flush_chunks();
                head_chunk = tail_chunk = chunk_id - (max_chunks - 1);
                head_pos = tail_pos = 0;
            }
            while (chunk_id >= tail_chunk)
            {
                if (std::size_t(tail_chunk - head_chunk) == max_chunks)
                    flush_head();
                chunks[tail_pos] = allocate(tail_chunk, place_data->batch_stats);
                if (chunks[tail_pos])
                {
                    chunks[tail_pos]->chunk_id = tail_chunk;
                    chunks[tail_pos]->stream_id = stream_id;
                }
                tail_chunk++;
                tail_pos++;
                if (tail_pos == max_chunks)
                    tail_pos = 0;  // wrap around circular buffer
            }
        }
        // Find position of chunk within the storage
        std::size_t pos = chunk_id - head_chunk + head_pos;
        if (pos >= max_chunks)
            pos -= max_chunks;  // wrap around the circular storage
        if (chunks[pos])
        {
            chunk &c = *chunks[pos];
            out.first = c.data.get() + place_data->heap_offset;
            metadata.chunk_id = chunk_id;
            metadata.heap_index = place_data->heap_index;
            metadata.heap_offset = place_data->heap_offset;
            metadata.chunk_ptr = &c;
            if (place_data->extra_size > 0)
            {
                assert(place_data->extra_size <= chunk_config.get_max_heap_extra());
                assert(c.extra);
                std::memcpy(c.extra.get() + place_data->extra_offset, place_data->extra, place_data->extra_size);
            }
            return out;
        }
        else
        {
            // the allocator didn't allocate a chunk for this slot.
            metadata.chunk_id = -1;
            metadata.chunk_ptr = nullptr;
            return out;
        }
    }
}

chunk_stream_allocator::chunk_stream_allocator(chunk_stream_state &stream)
    : stream(stream)
{
}

memory_allocator::pointer chunk_stream_allocator::allocate(std::size_t size, void *hint)
{
    if (hint)
    {
        auto alloc = stream.allocate(size, *reinterpret_cast<const packet_header *>(hint));
        // Use the heap_metadata as the deleter
        return pointer(alloc.first, std::move(alloc.second));
    }
    // Probably unreachable, but provides a safety net
    return memory_allocator::allocate(size, hint);
}

} // namespace detail

chunk_stream::chunk_stream(
    io_service_ref io_service,
    const stream_config &config,
    const chunk_stream_config &chunk_config)
    : chunk_stream_state(config, chunk_config),
    stream(std::move(io_service), adjust_config(config))
{
}

void chunk_stream::heap_ready(live_heap &&lh)
{
    if (lh.is_complete())
    {
        heap h(std::move(lh));
        auto metadata = get_heap_metadata(h.get_payload());
        // We need to check the chunk_id because the chunk might have been aged
        // out while the heap was incomplete.
        if (metadata && metadata->chunk_ptr && metadata->chunk_id >= get_head_chunk()
            && !get_chunk_config().get_packet_presence_payload_size())
        {
            assert(metadata->heap_index < metadata->chunk_ptr->present_size);
            metadata->chunk_ptr->present[metadata->heap_index] = true;
        }
    }
}

void chunk_stream::stop_received()
{
    stream::stop_received();
    flush_chunks();
}

void chunk_stream::stop()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        flush_chunks();
    }
    stream::stop();
}

chunk_stream::~chunk_stream()
{
    stop();
}

} // namespace recv
} // namespace spead2
