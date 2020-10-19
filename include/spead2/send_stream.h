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

#ifndef SPEAD2_SEND_STREAM_H
#define SPEAD2_SEND_STREAM_H

#include <functional>
#include <utility>
#include <vector>
#include <forward_list>
#include <memory>
#include <mutex>
#include <type_traits>
#include <future>
#include <atomic>
#include <cassert>
#include <boost/asio.hpp>
#include <boost/system/error_code.hpp>
#include <spead2/send_heap.h>
#include <spead2/common_defines.h>
#include <spead2/common_thread_pool.h>

namespace spead2
{

namespace send
{

enum class rate_method
{
    SW,        ///< Software rate limiter
    HW,        ///< Hardware rate limiter, if available
    AUTO       ///< Implementation decides on rate-limit method
};

/**
 * Configuration for send streams.
 */
class stream_config
{
public:
    static constexpr std::size_t default_max_packet_size = 1472;
    static constexpr std::size_t default_max_heaps = 4;
    static constexpr std::size_t default_burst_size = 65536;
    static constexpr double default_burst_rate_ratio = 1.05;
    static constexpr rate_method default_rate_method = rate_method::AUTO;

    /// Set maximum packet size to use (only counts the UDP payload, not L1-4 headers).
    stream_config &set_max_packet_size(std::size_t max_packet_size);
    /// Get maximum packet size to use.
    std::size_t get_max_packet_size() const { return max_packet_size; }
    /// Set maximum transmit rate to use, in bytes per second.
    stream_config &set_rate(double rate);
    /// Get maximum transmit rate to use, in bytes per second.
    double get_rate() const { return rate; }
    /// Set maximum size of a burst, in bytes.
    stream_config &set_burst_size(std::size_t burst_size);
    /// Get maximum size of a burst, in bytes.
    std::size_t get_burst_size() const { return burst_size; }
    /// Set maximum number of in-flight heaps.
    stream_config &set_max_heaps(std::size_t max_heaps);
    /// Get maximum number of in-flight heaps.
    std::size_t get_max_heaps() const { return max_heaps; }
    /// Set maximum increase in transmit rate for catching up.
    stream_config &set_burst_rate_ratio(double burst_rate_ratio);
    /// Get maximum increase in transmit rate for catching up.
    double get_burst_rate_ratio() const { return burst_rate_ratio; }
    /// Set rate-limiting method
    stream_config &set_rate_method(rate_method method);
    /// Get rate-limiting method
    rate_method get_rate_method() const { return method; }

    /// Get product of rate and burst_rate_ratio
    double get_burst_rate() const;

    stream_config();

private:
    std::size_t max_packet_size = default_max_packet_size;
    double rate = 0.0;
    std::size_t burst_size = default_burst_size;
    std::size_t max_heaps = default_max_heaps;
    double burst_rate_ratio = default_burst_rate_ratio;
    rate_method method = default_rate_method;
};

class writer;

/**
 * Stream for sending heaps, potentially to multiple destinations.
 */
class stream
{
public:
    typedef std::function<void(const boost::system::error_code &ec, item_pointer_t bytes_transferred)> completion_handler;

    /* Only public so that writer classes can update bytes_sent and result. */
    struct queue_item
    {
        const heap &h;
        item_pointer_t cnt;
        std::size_t substream_index;
        item_pointer_t bytes_sent = 0;
        boost::system::error_code result;
        completion_handler handler;
        // Populated by flush(). A forward_list takes less space when not used than vector.
        std::forward_list<std::promise<void>> waiters;

        queue_item() = default;
        queue_item(const heap &h, item_pointer_t cnt, std::size_t substream_index,
                   completion_handler &&handler) noexcept
            : h(h), cnt(cnt), substream_index(substream_index), handler(std::move(handler))
        {
        }
    };

private:
    friend class writer;

    typedef std::aligned_storage<sizeof(queue_item), alignof(queue_item)>::type queue_item_storage;

    /* Data are laid out in a manner designed to optimise the cache, which
     * means the logically related items (such as the head and tail indices)
     * might not be grouped together.
     */

    /* Data that's (mostly) read-only, and hence can be in shared state in both
     * caches.
     */

    /// Semantic size of the queue
    const std::size_t queue_size;
    /// Bitmask to be applied to queue indices for indexing
    const std::size_t queue_mask;
    /// Number of substreams exposed by the writer
    const std::size_t num_substreams;
    /// Increment to next_cnt after each heap
    item_pointer_t step_cnt = 1;
    /// Writer backing the stream
    std::unique_ptr<writer> w;
    /**
     * Circular queue with queue_mask + 1 slots. Items from @ref queue_head to
     * @ref queue_tail are constructed in place. Queue indices must be
     * interpreted mod queue_mask + 1 (a power of 2).
     */
    std::unique_ptr<queue_item_storage[]> queue;

    /* Data that's mostly written by the stream interface, but can be read by the
     * writer (however, when the queue empties, the writer will modify data).
     */

    /**
     * Protects access to
     * - writes to @ref queue_tail (but it may be *read* with atomic access)
     * - writes to the queue slot pointed to by @ref queue_tail
     * - @ref need_wakeup
     * - @ref next_cnt and @ref step_cnt
     *
     * The writer only needs the lock for @ref need_wakeup.
     */
    std::mutex tail_mutex;
    /// Position where next heap will be added
    std::atomic<std::size_t> queue_tail{0};
    /// Heap cnt for the next heap to send
    item_pointer_t next_cnt = 1;
    /// If true, the writer wants to be woken up when a new heap is added
    bool need_wakeup;

    /* Data that's only mostly written by the writer (apart from flush()), and
     * may be read by the stream.
     */

    /**
     * Protects access to
     * - writes to @ref queue_head (but it may be *read* with atomic access)
     * - the list of waiters in each queue item
     *
     * If taken together with @ref tail_mutex, @ref tail_mutex must be locked first.
     */
    std::mutex head_mutex;
    /// Oldest populated slot
    std::atomic<std::size_t> queue_head{0};

    /// Access an item from the queue (takes care of masking the index)
    queue_item *get_queue(std::size_t idx);

protected:
    writer &get_writer() { return *w; }
    const writer &get_writer() const { return *w; }

    explicit stream(std::unique_ptr<writer> &&w);

public:
    /// Retrieve the io_service used for processing the stream
    boost::asio::io_service &get_io_service() const;

    /**
     * Modify the linear sequence used to generate heap cnts. The next heap
     * will have cnt @a next, and each following cnt will be incremented by
     * @a step. When using this, it is the user's responsibility to ensure
     * that the generated values remain unique. The initial state is @a next =
     * 1, @a cnt = 1.
     *
     * This is useful when multiple senders will send heaps to the same
     * receiver, and need to keep their heap cnts separate.
     */
    void set_cnt_sequence(item_pointer_t next, item_pointer_t step);

    /**
     * Send @a h asynchronously, with @a handler called on completion. The
     * caller must ensure that @a h remains valid (as well as any memory it
     * points to) until @a handler is called.
     *
     * If this function returns @c true, then the heap has been added to the
     * queue. The completion handlers for such heaps are guaranteed to be
     * called in order.
     *
     * If this function returns @c false, the heap was rejected due to
     * insufficient space. The handler is called as soon as possible
     * (from a thread running the io_service), with error code @c
     * boost::asio::error::would_block.
     *
     * By default the heap cnt is chosen automatically (see @ref set_cnt_sequence).
     * An explicit value can instead be chosen by passing a non-negative value
     * for @a cnt. When doing this, it is entirely the responsibility of the
     * user to avoid collisions, both with other explicit values and with the
     * automatic counter. This feature is useful when multiple senders
     * contribute to a single stream and must keep their heap cnts disjoint,
     * which the automatic assignment would not do.
     *
     * Some streams may contain multiple substreams, each with a different
     * destination. In this case, @a substream_index selects the substream to
     * use.
     *
     * @retval  false  If the heap was immediately discarded
     * @retval  true   If the heap was enqueued
     */
    bool async_send_heap(const heap &h, completion_handler handler,
                         s_item_pointer_t cnt = -1,
                         std::size_t substream_index = 0);

    /**
     * Get the number of substreams in this stream.
     */
    std::size_t get_num_substreams() const;

    /**
     * Block until all enqueued heaps have been sent. This function is
     * thread-safe; only the heaps that were enqueued prior to calling the
     * function are waited for. The handlers will have been called prior
     * to this function returning.
     */
    void flush();

    virtual ~stream();
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_STREAM_H
