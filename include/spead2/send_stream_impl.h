/* Copyright 2019 SKA South Africa
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
 *
 * Contains the implementations from @ref spead2::send::stream_impl. It should
 * only be included by the .cpp files implementing its derived classes, to
 * reduce compile time.
 */

#ifndef SPEAD2_SEND_STREAM_IMPL_H
#define SPEAD2_SEND_STREAM_IMPL_H

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <boost/asio.hpp>
#include <boost/asio/high_resolution_timer.hpp>
#include <boost/utility/in_place_factory.hpp>
#include <spead2/common_defines.h>
#include <spead2/send_stream.h>

namespace spead2
{

namespace send
{

template<typename Derived, typename Extra>
std::size_t stream_impl<Derived, Extra>::next_queue_slot(std::size_t cur) const
{
    if (++cur == config.get_max_heaps() + 1)
        cur = 0;
    return cur;
}

template<typename Derived, typename Extra>
typename stream_impl<Derived, Extra>::queue_item *stream_impl<Derived, Extra>::get_queue(std::size_t idx)
{
    return reinterpret_cast<queue_item *>(queue.get() + idx);
}

template<typename Derived, typename Extra>
void stream_impl<Derived, Extra>::next_active()
{
    active = next_queue_slot(active);
    gen = boost::none;
}

template<typename Derived, typename Extra>
void stream_impl<Derived, Extra>::post_handler(boost::system::error_code result)
{
    queue_item &front = *get_queue(queue_head);
    get_io_service().post(
        std::bind(std::move(front.handler), result, front.bytes_sent));
    if (active == queue_head)
    {
        // Can only happen if there is an error with the head of the queue
        // before we've transmitted all its packets.
        assert(result);
        next_active();
    }
    front.~queue_item();
    queue_head = next_queue_slot(queue_head);
}

template<typename Derived, typename Extra>
bool stream_impl<Derived, Extra>::must_sleep() const
{
    return rate_bytes >= config.get_burst_size();
}

template<typename Derived, typename Extra>
void stream_impl<Derived, Extra>::process_results()
{
    for (std::size_t i = 0; i < n_current_packets; i++)
    {
        const transmit_packet &item = current_packets[i];
        if (item.item != get_queue(queue_head))
        {
            // A previous packet in this heap already aborted it
            continue;
        }
        if (item.result)
            post_handler(item.result);
        else
        {
            item.item->bytes_sent += item.size;
            if (item.last)
                post_handler(item.result);
        }
    }
    n_current_packets = 0;
}

template<typename Derived, typename Extra>
stream_impl<Derived, Extra>::timer_type::time_point stream_impl<Derived, Extra>::update_send_times(
    timer_type::time_point now)
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

template<typename Derived, typename Extra>
void stream_impl<Derived, Extra>::update_send_time_empty()
{
    timer_type::time_point now = timer_type::clock_type::now();
    // Compute what send_time would need to be to make the next packet due to be
    // transmitted now.
    std::chrono::duration<double> wait(rate_bytes * seconds_per_byte);
    auto wait2 = std::chrono::duration_cast<timer_type::clock_type::duration>(wait);
    timer_type::time_point backdate = now - wait2;
    send_time = std::max(send_time, backdate);
}

template<typename Derived, typename Extra>
void stream_impl<Derived, Extra>::load_packets(std::size_t tail)
{
    n_current_packets = 0;
    while (n_current_packets < max_current_packets && !must_sleep() && active != tail)
    {
        queue_item *cur = get_queue(active);
        if (!gen)
            gen = boost::in_place(cur->h, cur->cnt, config.get_max_packet_size());
        assert(gen->has_next_packet());
        transmit_packet &data = current_packets[n_current_packets];
        data.pkt = gen->next_packet();
        data.size = boost::asio::buffer_size(data.pkt.buffers);
        data.last = !gen->has_next_packet();
        data.item = cur;
        data.result = boost::system::error_code();
        rate_bytes += data.size;
        n_current_packets++;
        if (data.last)
            next_active();
    }
}

template<typename Derived, typename Extra>
void stream_impl<Derived, Extra>::do_next()
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    if (state == state_t::SENDING)
        process_results();
    else if (state == state_t::QUEUED)
        update_send_time_empty();
    assert(active == queue_head);

    if (must_sleep())
    {
        auto now = timer_type::clock_type::now();
        auto target_time = update_send_times(now);
        if (now < target_time)
        {
            state = state_t::SLEEPING;
            timer.expires_at(target_time);
            timer.async_wait([this](const boost::system::error_code &) { do_next(); });
            return;
        }
    }

    if (queue_head == queue_tail)
    {
        state = state_t::EMPTY;
        heap_empty.notify_all();
        return;
    }

    // Save a copy to use outside the protection of the lock.
    std::size_t tail = queue_tail;
    state = state_t::SENDING;
    lock.unlock();

    load_packets(tail);
    assert(n_current_packets > 0);
    static_cast<Derived *>(this)->async_send_packets();
}

template<typename Derived, typename Extra>
stream_impl<Derived, Extra>::stream_impl(
    io_service_ref io_service,
    const stream_config &config,
    std::size_t max_current_packets) :
        stream(std::move(io_service)),
        current_packets(new transmit_packet[max_current_packets]),
        max_current_packets(max_current_packets),
        config(config),
        seconds_per_byte_burst(config.get_burst_rate() > 0.0 ? 1.0 / config.get_burst_rate() : 0.0),
        seconds_per_byte(config.get_rate() > 0.0 ? 1.0 / config.get_rate() : 0.0),
        queue(new queue_item_storage[config.get_max_heaps() + 1]),
        timer(get_io_service())
{
}

template<typename Derived, typename Extra>
stream_impl<Derived, Extra>::~stream_impl()
{
    for (std::size_t i = queue_head; i != queue_tail; i = next_queue_slot(i))
        get_queue(i)->~queue_item();
}

template<typename Derived, typename Extra>
void stream_impl<Derived, Extra>::set_cnt_sequence(item_pointer_t next, item_pointer_t step)
{
    if (step == 0)
        throw std::invalid_argument("step cannot be 0");
    std::unique_lock<std::mutex> lock(queue_mutex);
    next_cnt = next;
    step_cnt = step;
}

template<typename Derived, typename Extra>
void stream_impl<Derived, Extra>::flush()
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    while (state != state_t::EMPTY)
    {
        heap_empty.wait(lock);
    }
}

template<typename Derived, typename Extra>
bool stream_impl<Derived, Extra>::async_send_heap(const heap &h, completion_handler handler, s_item_pointer_t cnt)
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    std::size_t new_tail = next_queue_slot(queue_tail);
    if (new_tail == queue_head)
    {
        lock.unlock();
        log_warning("async_send_heap: dropping heap because queue is full");
        get_io_service().post(std::bind(handler, boost::asio::error::would_block, 0));
        return false;
    }
    item_pointer_t cnt_mask = (item_pointer_t(1) << h.get_flavour().get_heap_address_bits()) - 1;
    if (cnt < 0)
    {
        cnt = next_cnt & cnt_mask;
        next_cnt += step_cnt;
    }
    else if (item_pointer_t(cnt) > cnt_mask)
    {
        lock.unlock();
        log_warning("async_send_heap: dropping heap because cnt is out of range");
        get_io_service().post(std::bind(handler, boost::asio::error::invalid_argument, 0));
        return false;
    }

    // Construct in place
    new (get_queue(queue_tail)) queue_item(h, cnt, std::move(handler));
    queue_tail = new_tail;

    bool empty = (state == state_t::EMPTY);
    if (empty)
        state = state_t::QUEUED;
    lock.unlock();

    if (empty)
        get_io_service().dispatch([this] { do_next(); });
    return true;
}

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_STREAM_IMPL_H
