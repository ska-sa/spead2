/* Copyright 2021-2023 National Research Foundation (SARAO)
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
#include <new>
#include <spead2/common_defines.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/recv_packet.h>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_heap.h>
#include <spead2/recv_utils.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_chunk_stream.h>

namespace spead2::recv
{

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

chunk_window::chunk_window(std::size_t max_chunks) : chunks(max_chunks) {}

chunk_stream_state_base::chunk_stream_state_base(
    const stream_config &config, const chunk_stream_config &chunk_config)
    : orig_memcpy(config.get_memcpy()),
    chunk_config(chunk_config),
    stream_id(config.get_stream_id()),
    base_stat_index(config.next_stat_index()),
    chunks(chunk_config.get_max_chunks())
{
    if (!this->chunk_config.get_place())
        throw std::invalid_argument("chunk_config.place is not set");

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
     * It's not clear whether it's actually portable in C++17: implementations
     * used to be allowed to add overhead for array new, even when using
     * placement new.  CWG 2382 disallowed that for placement new, and in
     * practice it sounds like no compiler ever added overhead for scalar types
     * (MSVC used to do it for polymorphic classes).
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

void chunk_stream_state_base::free_place_data::operator()(unsigned char *ptr) const
{
    // It's not totally clear whether std::launder is required here, but
    // better to be safe.
    auto *place_data = std::launder(reinterpret_cast<chunk_place_data *>(ptr));
    place_data->~chunk_place_data();
    operator delete(ptr);
}

void chunk_stream_state_base::packet_memcpy(
    const memory_allocator::pointer &allocation,
    const packet_header &packet) const
{
    const heap_metadata &metadata = *get_heap_metadata(allocation);
    if (chunk_too_old(metadata.chunk_id))
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

void chunk_stream_state_base::do_heap_ready(live_heap &&lh)
{
    if (lh.is_complete())
    {
        heap h(std::move(lh));
        auto metadata = get_heap_metadata(h.get_payload());
        // We need to check the chunk_id because the chunk might have been aged
        // out while the heap was incomplete.
        if (metadata && metadata->chunk_ptr
            && !chunk_too_old(metadata->chunk_id)
            && !get_chunk_config().get_packet_presence_payload_size())
        {
            assert(metadata->heap_index < metadata->chunk_ptr->present_size);
            metadata->chunk_ptr->present[metadata->heap_index] = true;
        }
    }
}

const chunk_stream_state_base::heap_metadata *chunk_stream_state_base::get_heap_metadata(
    const memory_allocator::pointer &ptr)
{
    return ptr.get_deleter().target<heap_metadata>();
}

chunk_manager_simple::chunk_manager_simple(const chunk_stream_config &chunk_config)
{
    if (!chunk_config.get_allocate())
        throw std::invalid_argument("chunk_config.allocate is not set");
    if (!chunk_config.get_ready())
        throw std::invalid_argument("chunk_config.ready is not set");
}

std::uint64_t *chunk_manager_simple::get_batch_stats(chunk_stream_state<chunk_manager_simple> &state) const
{
    return static_cast<chunk_stream *>(&state)->batch_stats.data();
}

chunk *chunk_manager_simple::allocate_chunk(chunk_stream_state<chunk_manager_simple> &state, std::int64_t chunk_id)
{
    const auto &allocate = state.chunk_config.get_allocate();
    std::unique_ptr<chunk> owned = allocate(chunk_id, state.place_data->batch_stats);
    return owned.release();  // ready_chunk will re-take ownership
}

void chunk_manager_simple::ready_chunk(chunk_stream_state<chunk_manager_simple> &state, chunk *c)
{
    std::unique_ptr<chunk> owned(c);
    state.chunk_config.get_ready()(std::move(owned), get_batch_stats(state));
}

template class chunk_stream_state<chunk_manager_simple>;
template class chunk_stream_allocator<chunk_manager_simple>;

} // namespace detail

chunk_stream::chunk_stream(
    io_service_ref io_service,
    const stream_config &config,
    const chunk_stream_config &chunk_config)
    : chunk_stream_state(config, chunk_config, detail::chunk_manager_simple(chunk_config)),
    stream(std::move(io_service), adjust_config(config))
{
}

void chunk_stream::heap_ready(live_heap &&lh)
{
    do_heap_ready(std::move(lh));
}

void chunk_stream::stop_received()
{
    stream::stop_received();
    flush_chunks();
}

void chunk_stream::stop()
{
    {
        std::lock_guard<std::mutex> lock(get_queue_mutex());
        flush_chunks();
    }
    stream::stop();
}

chunk_stream::~chunk_stream()
{
    stop();
}

} // namespace spead2::recv
