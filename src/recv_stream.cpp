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

#include <cstddef>
#include <utility>
#include <cassert>
#include <atomic>
#include <spead2/recv_stream.h>
#include <spead2/recv_live_heap.h>
#include <spead2/common_memcpy.h>
#include <spead2/common_thread_pool.h>

namespace spead2
{
namespace recv
{

constexpr std::size_t stream_base::default_max_heaps;

stream_base::stream_base(bug_compat_mask bug_compat, std::size_t max_heaps)
    : heap_storage(new storage_type[max_heaps]),
    heap_cnts(new s_item_pointer_t[max_heaps]),
    head(0),
    max_heaps(max_heaps), bug_compat(bug_compat),
    allocator(std::make_shared<memory_allocator>())
{
    for (std::size_t i = 0; i < max_heaps; i++)
        heap_cnts[i] = -1;
}

stream_base::~stream_base()
{
    for (std::size_t i = 0; i < max_heaps; i++)
        if (heap_cnts[i] != -1)
            reinterpret_cast<live_heap *>(&heap_storage[i])->~live_heap();
}

void stream_base::set_memory_pool(std::shared_ptr<memory_pool> pool)
{
    set_memory_allocator(std::move(pool));
}

void stream_base::set_memory_allocator(std::shared_ptr<memory_allocator> allocator)
{
    std::lock_guard<std::mutex> lock(allocator_mutex);
    this->allocator = std::move(allocator);
}

void stream_base::set_memcpy(memcpy_function memcpy)
{
    this->memcpy.store(memcpy, std::memory_order_relaxed);
}

void stream_base::set_memcpy(memcpy_function_id id)
{
    switch (id)
    {
    case MEMCPY_STD:
        set_memcpy(std::memcpy);
        break;
    case MEMCPY_NONTEMPORAL:
        set_memcpy(spead2::memcpy_nontemporal);
        break;
    default:
        throw std::invalid_argument("Unknown memcpy function");
    }
}

bool stream_base::add_packet(const packet_header &packet)
{
    assert(!stopped);
    // Look for matching heap. For large heaps, this will in most
    // cases be in the head position.
    live_heap *h = NULL;
    std::size_t position = 0;
    s_item_pointer_t heap_cnt = packet.heap_cnt;
    if (heap_cnts[head] == heap_cnt)
    {
        position = head;
        h = reinterpret_cast<live_heap *>(&heap_storage[head]);
    }
    else
    {
        for (std::size_t i = 0; i < max_heaps; i++)
            if (heap_cnts[i] == heap_cnt)
            {
                position = i;
                h = reinterpret_cast<live_heap *>(&heap_storage[i]);
                break;
            }

        if (!h)
        {
            // Never seen this heap before. Evict the old one in its slot,
            // if any. Note: not safe to dereference h just anywhere here!
            if (++head == max_heaps)
                head = 0;
            position = head;
            h = reinterpret_cast<live_heap *>(&heap_storage[head]);
            if (heap_cnts[head] != -1)
            {
                heap_ready(std::move(*h));
                h->~live_heap();
            }
            heap_cnts[head] = heap_cnt;
            std::shared_ptr<memory_allocator> allocator;
            {
                std::lock_guard<std::mutex> lock(allocator_mutex);
                allocator = this->allocator;
            }
            new (h) live_heap(heap_cnt, bug_compat, allocator);
            h->set_memcpy(memcpy.load(std::memory_order_relaxed));
        }
    }

    bool result = false;
    bool end_of_stream = false;
    if (h->add_packet(packet))
    {
        result = true;
        end_of_stream = h->is_end_of_stream();
        if (h->is_complete())
        {
            if (!end_of_stream)
                heap_ready(std::move(*h));
            heap_cnts[position] = -1;
            h->~live_heap();
        }
    }

    if (end_of_stream)
        stop_received();
    return result;
}

void stream_base::flush()
{
    for (std::size_t i = 0; i < max_heaps; i++)
    {
        if (++head == max_heaps)
            head = 0;
        if (heap_cnts[head] != -1)
        {
            live_heap *h = reinterpret_cast<live_heap *>(&heap_storage[head]);
            heap_ready(std::move(*h));
            h->~live_heap();
            heap_cnts[head] = -1;
        }
    }
}

void stream_base::stop_received()
{
    stopped = true;
    flush();
}


stream::stream(boost::asio::io_service &io_service, bug_compat_mask bug_compat, std::size_t max_heaps)
    : stream_base(bug_compat, max_heaps), strand(io_service)
{
}

stream::stream(thread_pool &thread_pool, bug_compat_mask bug_compat, std::size_t max_heaps)
    : stream(thread_pool.get_io_service(), bug_compat, max_heaps)
{
}

void stream::stop_received()
{
    // Check for already stopped, so that readers are stopped exactly once
    if (!is_stopped())
    {
        stream_base::stop_received();
        for (const auto &reader : readers)
            reader->stop();
    }
}

void stream::stop_impl()
{
    run_in_strand([this] { stop_received(); });

    /* Block until all readers have entered their final completion handler.
     * Note that this cannot conflict with a previously issued emplace_reader,
     * because its emplace_reader_callback either happens-before the
     * stop_received call above or sees that the stream has been stopped and
     * does not touch the readers list.
     */
    for (const auto &r : readers)
        r->join();

    // Destroy the readers with the strand held, to ensure that the
    // completion handlers have actually returned.
    run_in_strand([this] { readers.clear(); });
}

void stream::stop()
{
    std::call_once(stop_once, [this] { stop_impl(); });
}

stream::~stream()
{
    stop();
}


const std::uint8_t *mem_to_stream(stream_base &s, const std::uint8_t *ptr, std::size_t length)
{
    while (length > 0 && !s.is_stopped())
    {
        packet_header packet;
        std::size_t size = decode_packet(packet, ptr, length);
        if (size > 0)
        {
            s.add_packet(packet);
            ptr += size;
            length -= size;
        }
        else
            length = 0; // causes loop to exit
    }
    return ptr;
}

} // namespace recv
} // namespace spead2
