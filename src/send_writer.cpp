/* Copyright 2020 National Research Foundation (SARAO)
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
#include <array>
#include <chrono>
#include <forward_list>
#include <algorithm>
#include <boost/utility/in_place_factory.hpp>
#include <spead2/send_writer.h>
#include <spead2/send_stream.h>

namespace spead2
{
namespace send
{

void writer::set_owner(stream *owner)
{
    assert(!this->owner);
    assert(owner);
    this->owner = owner;
}

void writer::enable_hw_rate()
{
    assert(config.get_rate_method() != rate_method::SW);
    hw_rate = true;
}

writer::timer_type::time_point writer::update_send_times(timer_type::time_point now)
{
    std::chrono::duration<double> wait_burst(rate_bytes * seconds_per_byte_burst);
    std::chrono::duration<double> wait(rate_bytes * seconds_per_byte);
    send_time_burst += std::chrono::duration_cast<timer_type::clock_type::duration>(wait_burst);
    send_time += std::chrono::duration_cast<timer_type::clock_type::duration>(wait);
    rate_bytes = 0;

    /* send_time_burst needs to reflect the time the burst
     * was actually sent (as well as we can estimate it), even if
     * send_time or now is later.
     */
    timer_type::time_point target_time = std::max(send_time_burst, send_time);
    send_time_burst = std::max(now, target_time);
    return target_time;
}

void writer::update_send_time_empty()
{
    timer_type::time_point now = timer_type::clock_type::now();
    // Compute what send_time would need to be to make the next packet due to be
    // transmitted now.
    std::chrono::duration<double> wait(rate_bytes * seconds_per_byte);
    auto wait2 = std::chrono::duration_cast<timer_type::clock_type::duration>(wait);
    timer_type::time_point backdate = now - wait2;
    send_time = std::max(send_time, backdate);
}

writer::packet_result writer::get_packet(transmit_packet &data, std::uint8_t *scratch)
{
    if (must_sleep)
    {
        auto now = timer_type::clock_type::now();
        if (now < send_time_burst)
            return packet_result::SLEEP;
        else
            must_sleep = false;
    }
    if (rate_bytes >= config.get_burst_size())
    {
        auto now = timer_type::clock_type::now();
        auto target_time = update_send_times(now);
        if (now < target_time)
        {
            must_sleep = true;
            return packet_result::SLEEP;
        }
    }

    if (active == queue_tail)
    {
        /* We've read up to our cached tail. See if there has been
         * new work added in the meantime.
         */
        queue_tail = get_owner()->queue_tail.load(std::memory_order_acquire);
        if (active == queue_tail)
            return packet_result::EMPTY;
    }
    detail::queue_item *cur = get_owner()->get_queue(active);
    assert(cur->gen.has_next_packet());

    data.buffers = cur->gen.next_packet(scratch);
    data.size = boost::asio::buffer_size(data.buffers);
    data.substream_index = cur->substream_index;
    // Point at the start of the group, so that errors and byte counts accumulate
    // in one place.
    data.item = get_owner()->get_queue(active_start);
    if (!hw_rate)
        rate_bytes += data.size;
    data.last = false;

    // Find the heap to use for the next packet, skipping exhausted heaps
    std::size_t next_active = cur->group_next;
    detail::queue_item *next = (next_active == active) ? cur : get_owner()->get_queue(next_active);
    while (!next->gen.has_next_packet())
    {
        if (next_active == active)
        {
            // We've gone all the way around the group and not found anything,
            // so the group is exhausted.
            data.last = true;
            active = cur->group_end;
            active_start = active;
            return packet_result::SUCCESS;
        }
        next_active = next->group_next;
        next = get_owner()->get_queue(next_active);
    }
    // Cache the result so that we can skip the search next time
    cur->group_next = next_active;
    active = next_active;
    return packet_result::SUCCESS;
}

void writer::groups_completed(std::size_t n)
{
    struct bound_handler
    {
        stream::completion_handler handler;
        boost::system::error_code result;
        std::size_t bytes_transferred;

        bound_handler() = default;
        bound_handler(stream::completion_handler &&handler, const boost::system::error_code &result, std::size_t bytes_transferred)
            : handler(std::move(handler)), result(result), bytes_transferred(bytes_transferred)
        {
        }
    };

    /**
     * We have to ensure that we vacate the slots (and update the head)
     * before we trigger any callbacks or promises, because otherwise a
     * callback that immediately tries to enqueue new heaps might find that
     * the queue is still full.
     *
     * Batching amortises the locking overhead when using small heaps. We
     * could use a single dynamically-sized batch, but that would require
     * dynamic memory allocation to hold the handlers.
     */
    constexpr std::size_t max_batch = 16;
    std::forward_list<std::promise<void>> waiters;
    std::array<bound_handler, max_batch> handlers;
    while (n > 0)
    {
        std::size_t batch = std::min(max_batch, n);
        {
            std::lock_guard<std::mutex> lock(get_owner()->head_mutex);
            for (std::size_t i = 0; i < batch; i++)
            {
                detail::queue_item *cur = get_owner()->get_queue(queue_head);
                handlers[i] = bound_handler(
                    std::move(cur->handler), cur->result, cur->bytes_sent);
                waiters.splice_after(waiters.before_begin(), cur->waiters);
                std::size_t next_queue_head = cur->group_end;
                cur->~queue_item();
                queue_head++;
                // For a group with > 1 heap, destroy the rest of the group
                // and splice in waiters.
                while (queue_head != next_queue_head)
                {
                    cur = get_owner()->get_queue(queue_head);
                    waiters.splice_after(waiters.before_begin(), cur->waiters);
                    cur->~queue_item();
                    queue_head++;
                }
            }
            // After this, async_send_heaps is free to reuse the slots we've
            // just vacated.
            get_owner()->queue_head.store(queue_head, std::memory_order_release);
        }
        for (std::size_t i = 0; i < batch; i++)
            handlers[i].handler(handlers[i].result, handlers[i].bytes_transferred);
        while (!waiters.empty())
        {
            waiters.front().set_value();
            waiters.pop_front();
        }
        n -= batch;
    }
}

void writer::sleep()
{
    if (must_sleep)
    {
        timer.expires_at(send_time_burst);
        timer.async_wait(
            [this](const boost::system::error_code &) {
                must_sleep = false;
                wakeup();
            }
        );
    }
    else
    {
        post_wakeup();
    }
}

void writer::request_wakeup()
{
    std::size_t old_tail = queue_tail;
    {
        std::lock_guard<std::mutex> tail_lock(get_owner()->tail_mutex);
        queue_tail = get_owner()->queue_tail.load(std::memory_order_acquire);
        if (queue_tail == old_tail)
        {
            get_owner()->need_wakeup = true;
            return;
        }
    }
    // If we get here, new work was added since the last call to get_packet,
    // so we must just wake ourselves up.
    post_wakeup();
}

void writer::post_wakeup()
{
    get_io_service().post([this]() { wakeup(); });
}

writer::writer(io_service_ref io_service, const stream_config &config)
    : config(config),
    seconds_per_byte_burst(config.get_burst_rate() > 0.0 ? 1.0 / config.get_burst_rate() : 0.0),
    seconds_per_byte(config.get_rate() > 0.0 ? 1.0 / config.get_rate() : 0.0),
    io_service(std::move(io_service)),
    timer(*this->io_service)
{
}

} // namespace send
} // namespace spead2
