/* Copyright 2015, 2017, 2019 SKA South Africa
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

#include <cmath>
#include <stdexcept>
#include <spead2/send_stream.h>

namespace spead2
{
namespace send
{

constexpr std::size_t stream_config::default_max_packet_size;
constexpr std::size_t stream_config::default_max_heaps;
constexpr std::size_t stream_config::default_burst_size;
constexpr double stream_config::default_burst_rate_ratio;

void stream_config::set_max_packet_size(std::size_t max_packet_size)
{
    // TODO: validate here instead rather than waiting until packet_generator
    this->max_packet_size = max_packet_size;
}

std::size_t stream_config::get_max_packet_size() const
{
    return max_packet_size;
}

void stream_config::set_rate(double rate)
{
    if (rate < 0.0 || !std::isfinite(rate))
        throw std::invalid_argument("rate must be non-negative");
    this->rate = rate;
}

double stream_config::get_rate() const
{
    return rate;
}

void stream_config::set_max_heaps(std::size_t max_heaps)
{
    if (max_heaps == 0)
        throw std::invalid_argument("max_heaps must be positive");
    this->max_heaps = max_heaps;
}

std::size_t stream_config::get_max_heaps() const
{
    return max_heaps;
}

void stream_config::set_burst_size(std::size_t burst_size)
{
    this->burst_size = burst_size;
}

std::size_t stream_config::get_burst_size() const
{
    return burst_size;
}

void stream_config::set_burst_rate_ratio(double burst_rate_ratio)
{
    if (burst_rate_ratio < 1.0 || !std::isfinite(burst_rate_ratio))
        throw std::invalid_argument("burst rate ratio must be at least 1.0 and finite");
    this->burst_rate_ratio = burst_rate_ratio;
}

double stream_config::get_burst_rate_ratio() const
{
    return burst_rate_ratio;
}

double stream_config::get_burst_rate() const
{
    return rate * burst_rate_ratio;
}

stream_config::stream_config(
    std::size_t max_packet_size,
    double rate,
    std::size_t burst_size,
    std::size_t max_heaps,
    double burst_rate_ratio)
{
    set_max_packet_size(max_packet_size);
    set_rate(rate);
    set_burst_size(burst_size);
    set_max_heaps(max_heaps);
    set_burst_rate_ratio(burst_rate_ratio);
}


stream::stream(io_service_ref io_service)
    : io_service(std::move(io_service))
{
}

stream::~stream()
{
}


std::size_t stream_impl_base::next_queue_slot(std::size_t cur) const
{
    if (++cur == config.get_max_heaps() + 1)
        cur = 0;
    return cur;
}

stream_impl_base::queue_item *stream_impl_base::get_queue(std::size_t idx)
{
    return reinterpret_cast<queue_item *>(queue.get() + idx);
}

void stream_impl_base::next_active()
{
    active = next_queue_slot(active);
    gen = boost::none;
}

void stream_impl_base::post_handler(boost::system::error_code result)
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

bool stream_impl_base::must_sleep() const
{
    return rate_bytes >= config.get_burst_size();
}

void stream_impl_base::process_results()
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

stream_impl_base::timer_type::time_point stream_impl_base::update_send_times(
    timer_type::time_point now)
{
    std::chrono::duration<double> wait_burst(rate_bytes * seconds_per_byte_burst);
    std::chrono::duration<double> wait(rate_bytes * seconds_per_byte);
    send_time_burst += std::chrono::duration_cast<timer_type::clock_type::duration>(wait_burst);
    send_time += std::chrono::duration_cast<timer_type::clock_type::duration>(wait);
    rate_bytes = 0;

    /* send_time_burst needs to reflect the time the burst
     * was actually sent (as well as we can estimate it), even if
     * sent_time or now is later.
     */
    timer_type::time_point target_time = std::max(send_time_burst, send_time);
    send_time_burst = std::max(now, target_time);
    return target_time;
}

void stream_impl_base::update_send_time_empty()
{
    timer_type::time_point now = timer_type::clock_type::now();
    // Compute what send_time would need to be to make the next packet due to be
    // transmitted now.
    std::chrono::duration<double> wait(rate_bytes * seconds_per_byte);
    auto wait2 = std::chrono::duration_cast<timer_type::clock_type::duration>(wait);
    timer_type::time_point backdate = now - wait2;
    send_time = std::max(send_time, backdate);
}

void stream_impl_base::load_packets(std::size_t tail)
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

stream_impl_base::stream_impl_base(
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

stream_impl_base::~stream_impl_base()
{
    for (std::size_t i = queue_head; i != queue_tail; i = next_queue_slot(i))
        get_queue(i)->~queue_item();
}

void stream_impl_base::set_cnt_sequence(item_pointer_t next, item_pointer_t step)
{
    if (step == 0)
        throw std::invalid_argument("step cannot be 0");
    std::unique_lock<std::mutex> lock(queue_mutex);
    next_cnt = next;
    step_cnt = step;
}

void stream_impl_base::flush()
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    while (state != state_t::EMPTY)
    {
        heap_empty.wait(lock);
    }
}

} // namespace send
} // namespace spead2
