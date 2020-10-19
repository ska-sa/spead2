/* Copyright 2015, 2017, 2019-2020 National Research Foundation (SARAO)
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

#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <spead2/common_logging.h>
#include <spead2/send_stream.h>
#include <spead2/send_writer.h>

namespace spead2
{
namespace send
{

constexpr std::size_t stream_config::default_max_packet_size;
constexpr std::size_t stream_config::default_max_heaps;
constexpr std::size_t stream_config::default_burst_size;
constexpr double stream_config::default_burst_rate_ratio;
constexpr rate_method stream_config::default_rate_method;

stream_config &stream_config::set_max_packet_size(std::size_t max_packet_size)
{
    // TODO: validate here instead rather than waiting until packet_generator
    this->max_packet_size = max_packet_size;
    return *this;
}

stream_config &stream_config::set_rate(double rate)
{
    if (rate < 0.0 || !std::isfinite(rate))
        throw std::invalid_argument("rate must be non-negative");
    this->rate = rate;
    return *this;
}

stream_config &stream_config::set_max_heaps(std::size_t max_heaps)
{
    if (max_heaps == 0)
        throw std::invalid_argument("max_heaps must be positive");
    this->max_heaps = max_heaps;
    return *this;
}

stream_config &stream_config::set_burst_size(std::size_t burst_size)
{
    this->burst_size = burst_size;
    return *this;
}

stream_config &stream_config::set_burst_rate_ratio(double burst_rate_ratio)
{
    if (burst_rate_ratio < 1.0 || !std::isfinite(burst_rate_ratio))
        throw std::invalid_argument("burst rate ratio must be at least 1.0 and finite");
    this->burst_rate_ratio = burst_rate_ratio;
    return *this;
}

stream_config &stream_config::set_rate_method(rate_method method)
{
    this->method = method;
    return *this;
}

double stream_config::get_burst_rate() const
{
    return rate * burst_rate_ratio;
}

stream_config::stream_config()
{
}


stream::queue_item *stream::get_queue(std::size_t idx)
{
    return reinterpret_cast<queue_item *>(queue.get() + (idx & queue_mask));
}

static std::size_t compute_queue_mask(std::size_t size)
{
    if (size == 0)
        throw std::invalid_argument("max_heaps must be at least 1");
    if (size > std::numeric_limits<std::size_t>::max() / 2 + 1)
        throw std::invalid_argument("max_heaps is too large");
    std::size_t p2 = 1;
    while (p2 < size)
        p2 <<= 1;
    return p2 - 1;
}

stream::stream(std::unique_ptr<writer> &&w)
    : queue_size(w->config.get_max_heaps()),
    queue_mask(compute_queue_mask(queue_size)),
    num_substreams(w->get_num_substreams()),
    w(std::move(w)),
    queue(new queue_item_storage[queue_mask + 1])
{
    this->w->set_owner(this);
    this->w->start();
}

stream::~stream()
{
    flush();
}

boost::asio::io_service &stream::get_io_service() const
{
    return w->get_io_service();
}

void stream::set_cnt_sequence(item_pointer_t next, item_pointer_t step)
{
    if (step == 0)
        throw std::invalid_argument("step cannot be 0");
    std::unique_lock<std::mutex> lock(tail_mutex);
    next_cnt = next;
    step_cnt = step;
}

bool stream::async_send_heap(const heap &h, completion_handler handler,
                             s_item_pointer_t cnt,
                             std::size_t substream_index)
{
    if (substream_index >= num_substreams)
    {
        log_warning("async_send_heap: dropping heap because substream index is out of range");
        get_io_service().post(std::bind(handler, boost::asio::error::invalid_argument, 0));
        return false;
    }
    item_pointer_t cnt_mask = (item_pointer_t(1) << h.get_flavour().get_heap_address_bits()) - 1;

    std::unique_lock<std::mutex> lock(head_mutex);
    std::size_t tail = queue_tail.load(std::memory_order_relaxed);
    std::size_t head = queue_head.load(std::memory_order_acquire);
    if (tail - head == queue_size)
    {
        lock.unlock();
        log_warning("async_send_heap: dropping heap because queue is full");
        get_io_service().post(std::bind(handler, boost::asio::error::would_block, 0));
        return false;
    }
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
    new (get_queue(tail)) queue_item(h, cnt, substream_index, std::move(handler));
    bool wakeup = need_wakeup;
    need_wakeup = false;
    queue_tail.store(tail + 1, std::memory_order_release);
    lock.unlock();

    if (wakeup)
    {
        writer *w_ptr = w.get();
        get_io_service().post([w_ptr]() {
            w_ptr->update_send_time_empty();
            w_ptr->wakeup();
        });
    }
    return true;
}

void stream::flush()
{
    std::future<void> future;
    {
        std::lock_guard<std::mutex> tail_lock(tail_mutex);
        std::lock_guard<std::mutex> head_lock(head_mutex);
        // These could probably be read with relaxed consistency because the locks
        // ensure ordering, but this is not performance-critical.
        std::size_t tail = queue_tail.load();
        std::size_t head = queue_head.load();
        if (head == tail)
            return;
        queue_item *item = get_queue(tail - 1);
        item->waiters.emplace_front();
        future = item->waiters.front().get_future();
    }

    future.wait();
}

std::size_t stream::get_num_substreams() const
{
    return num_substreams;
}

} // namespace send
} // namespace spead2
