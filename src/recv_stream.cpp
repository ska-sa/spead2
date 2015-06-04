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
#include "recv_stream.h"
#include "recv_live_heap.h"
#include "common_thread_pool.h"

namespace spead2
{
namespace recv
{

constexpr std::size_t stream_base::default_max_heaps;

stream_base::stream_base(bug_compat_mask bug_compat, std::size_t max_heaps)
    : max_heaps(max_heaps), bug_compat(bug_compat)
{
}

void stream_base::set_max_heaps(std::size_t max_heaps)
{
    this->max_heaps = max_heaps;
}

void stream_base::set_memory_pool(std::shared_ptr<memory_pool> pool)
{
    this->pool = std::move(pool);
}

bool stream_base::add_packet(const packet_header &packet)
{
    assert(!stopped);
    // Look for matching heap
    auto insert_before = heaps.begin();
    bool result = false;
    bool end_of_stream = false;
    bool found = false;
    for (auto it = heaps.begin(); it != heaps.end(); ++it)
    {
        live_heap &h = *it;
        if (h.get_cnt() == packet.heap_cnt)
        {
            found = true;
            if (h.add_packet(packet))
            {
                result = true;
                end_of_stream = h.is_end_of_stream();
                if (h.is_complete())
                {
                    heap_ready(std::move(h));
                    heaps.erase(it);
                }
            }
            break;
        }
        else if (h.get_cnt() < packet.heap_cnt)
            insert_before = next(it);
    }

    if (!found)
    {
        // Doesn't match any previously seen heap, so create a new one
        live_heap h(packet.heap_cnt, bug_compat);
        h.set_memory_pool(pool);
        if (h.add_packet(packet))
        {
            result = true;
            end_of_stream = h.is_end_of_stream();
            if (h.is_complete())
            {
                heap_ready(std::move(h));
            }
            else
            {
                heaps.insert(insert_before, std::move(h));
                if (heaps.size() > max_heaps)
                {
                    // Too many active heaps: pop the lowest ID, even if incomplete
                    heap_ready(std::move(heaps[0]));
                    heaps.pop_front();
                }
            }
        }
    }
    if (end_of_stream)
        stop_received();
    return result;
}

void stream_base::flush()
{
    for (live_heap &h : heaps)
    {
        heap_ready(std::move(h));
    }
    heaps.clear();
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

void stream::set_max_heaps(std::size_t max_heaps)
{
    run_in_strand([this, max_heaps] { stream_base::set_max_heaps(max_heaps); });
}

void stream::set_memory_pool(std::shared_ptr<memory_pool> pool)
{
    run_in_strand([this, pool] { stream_base::set_memory_pool(std::move(pool)); });
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
