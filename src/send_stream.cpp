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
#include <thread>
#include <stdexcept>
#include <spead2/common_logging.h>
#include <spead2/send_stream.h>
#include <spead2/send_writer.h>

namespace spead2
{
namespace send
{

detail::queue_item *stream::get_queue(std::size_t idx)
{
    return reinterpret_cast<detail::queue_item *>(queue.get() + (idx & queue_mask));
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

stream::unwinder::unwinder(stream &s, std::size_t tail)
    : s(s), orig_tail(tail), tail(tail)
{
}

void stream::unwinder::set_tail(std::size_t tail)
{
    this->tail = tail;
}

void stream::unwinder::abort()
{
    for (std::size_t i = orig_tail; i != tail; i++)
        s.get_queue(i)->~queue_item();
}

void stream::unwinder::commit()
{
    orig_tail = tail;
}

stream::stream(std::unique_ptr<writer> &&w)
    : queue_size(w->config.get_max_heaps()),
    queue_mask(compute_queue_mask(queue_size)),
    num_substreams(w->get_num_substreams()),
    max_packet_size(w->config.get_max_packet_size()),
    w(std::move(w)),
    queue(new queue_item_storage[queue_mask + 1])
{
    this->w->set_owner(this);
    this->w->start();
}

stream::~stream()
{
    flush();
    /* The writer might still have a pending wakeup to check for new work.
     * Before we can safely delete it, we need it to have set need_wakeup.
     * A spin loop is not normally great style, but we take a hit on shutdown
     * to keep worker::request_wakeup fast when we're not shutting down.
     */
    std::unique_lock<std::mutex> lock(tail_mutex);
    while (!need_wakeup)
    {
        lock.unlock();
        std::this_thread::yield();
        lock.lock();
    }
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
    heap_reference ref(h, cnt, substream_index);
    return async_send_heaps_impl<null_unwinder>(
        &ref, &ref + 1, std::move(handler), group_mode::ROUND_ROBIN);
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
        detail::queue_item *item = get_queue(tail - 1);
        item->waiters.emplace_front();
        future = item->waiters.front().get_future();
    }

    future.wait();
}

std::size_t stream::get_num_substreams() const
{
    return num_substreams;
}

// Explicit instantiation
template bool stream::async_send_heaps_impl<stream::null_unwinder, heap_reference *>(
    heap_reference *first, heap_reference *last,
    completion_handler &&handler, group_mode mode);

} // namespace send
} // namespace spead2
