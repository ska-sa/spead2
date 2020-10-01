/* Copyright 2015, 2017, 2019-2020 SKA South Africa
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
#include <spead2/send_stream.h>

namespace spead2
{
namespace send
{

constexpr std::size_t stream_config::default_max_packet_size;
constexpr std::size_t stream_config::default_max_heaps;
constexpr std::size_t stream_config::default_burst_size;
constexpr double stream_config::default_burst_rate_ratio;
constexpr bool stream_config::default_allow_hw_rate;

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

stream_config &stream_config::set_allow_hw_rate(bool allow_hw_rate)
{
    this->allow_hw_rate = allow_hw_rate;
    return *this;
}

double stream_config::get_burst_rate() const
{
    return rate * burst_rate_ratio;
}

stream_config::stream_config()
{
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
     * send_time or now is later.
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
        if (!hw_rate)
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

void stream_impl_base::enable_hw_rate()
{
    assert(config.get_allow_hw_rate());
    hw_rate = true;
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

///////////////////////// TODO boundary with old code

void writer::set_owner(stream2 *owner)
{
    assert(!this->owner);
    assert(owner);
    this->owner = owner;
}

void writer::enable_hw_rate()
{
    assert(config.get_allow_hw_rate());
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

writer::packet_result writer::get_packet(transmit_packet &data)
{
    if (must_sleep)
        return packet_result::SLEEP;
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
    stream2::queue_item *cur = get_owner()->get_queue(active);
    if (!gen)
        gen = boost::in_place(cur->h, cur->cnt, config.get_max_packet_size());
    assert(gen->has_next_packet());

    data.pkt = gen->next_packet();
    data.size = boost::asio::buffer_size(data.pkt.buffers);
    data.last = !gen->has_next_packet();
    data.item = cur;
    if (!hw_rate)
        rate_bytes += data.size;
    if (data.last)
    {
        active++;
        gen = boost::none;
    }
    return packet_result::SUCCESS;
}

void writer::heaps_completed(std::size_t n)
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
     * could a single dynamically-sized batch, but that would require
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
                stream2::queue_item *cur = get_owner()->get_queue(queue_head);
                handlers[i] = bound_handler(
                    std::move(cur->handler), cur->result, cur->bytes_sent);
                waiters.splice_after(waiters.before_begin(), cur->waiters);
                cur->~queue_item();
                queue_head++;
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
        });
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

stream2::queue_item *stream2::get_queue(std::size_t idx)
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

// TODO: eliminate io_service from stream base class
stream2::stream2(std::unique_ptr<writer> &&w)
    : stream(w->get_io_service()),
    queue_size(w->config.get_max_heaps()),
    queue_mask(compute_queue_mask(queue_size)),
    num_substreams(w->get_num_substreams()),
    w(std::move(w)),
    queue(new queue_item_storage[queue_mask + 1])
{
    this->w->set_owner(this);
    this->w->start();
}

stream2::~stream2()
{
    flush();
}

void stream2::set_cnt_sequence(item_pointer_t next, item_pointer_t step)
{
    if (step == 0)
        throw std::invalid_argument("step cannot be 0");
    std::unique_lock<std::mutex> lock(tail_mutex);
    next_cnt = next;
    step_cnt = step;
}

bool stream2::async_send_heap(const heap &h, completion_handler handler,
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

void stream2::flush()
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

std::size_t stream2::get_num_substreams() const
{
    return num_substreams;
}

} // namespace send
} // namespace spead2
