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

#ifndef SPEAD2_SEND_STREAM_H
#define SPEAD2_SEND_STREAM_H

#include <functional>
#include <utility>
#include <vector>
#include <forward_list>
#include <memory>
#include <chrono>
#include <mutex>
#include <iterator>
#include <type_traits>
#include <condition_variable>
#include <future>
#include <stdexcept>
#include <cassert>
#include <boost/asio.hpp>
#include <boost/asio/high_resolution_timer.hpp>
#include <boost/system/error_code.hpp>
#include <boost/optional.hpp>
#include <boost/utility/in_place_factory.hpp>
#include <spead2/send_heap.h>
#include <spead2/send_packet.h>
#include <spead2/common_logging.h>
#include <spead2/common_defines.h>
#include <spead2/common_thread_pool.h>

namespace spead2
{

namespace send
{

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
    static constexpr bool default_allow_hw_rate = true;

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
    /// Set whether to allow hardware rate limiting.
    stream_config &set_allow_hw_rate(bool allow_hw_rate);
    /// Get whether to allow hardware rate limiting.
    bool get_allow_hw_rate() const { return allow_hw_rate; }

    /// Get product of rate and burst_rate_ratio
    double get_burst_rate() const;

    stream_config();

private:
    std::size_t max_packet_size = default_max_packet_size;
    double rate = 0.0;
    std::size_t burst_size = default_burst_size;
    std::size_t max_heaps = default_max_heaps;
    double burst_rate_ratio = default_burst_rate_ratio;
    bool allow_hw_rate = default_allow_hw_rate;
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

    // Padding to ensure that the above doesn't share a cache line with the below
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif
    std::uint8_t padding1[64];
#ifdef __clang__
#pragma clang diagnostic pop
#endif

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

    // Padding to ensure that the above doesn't share a cache line with the below
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif
    std::uint8_t padding2[64];
#ifdef __clang__
#pragma clang diagnostic pop
#endif

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
     * thread-safe, but can be live-locked if more heaps are added while it is
     * running.
     */
    void flush();

    virtual ~stream();
};

// TODO: move to separate send_writer.h
class writer
{
protected:
    enum class packet_result
    {
        /**
         * A new packet has been returned.
         */
        SUCCESS,
        /**
         * No packet because we need to sleep for rate limiting. Use
         * @ref sleep to request that @ref wakeup be called when it is
         * time to resume. Until that's done, @ref get_packet will continue
         * to return @c SLEEP.
         */
        SLEEP,
        /**
         * There are no more packets currently available. Use @ref
         * request_wakeup to ask to be woken when a new heap is added.
         */
        EMPTY
    };

private:
    friend class stream;

    typedef boost::asio::basic_waitable_timer<std::chrono::high_resolution_clock> timer_type;

    const stream_config config;    // TODO: probably doesn't need the whole thing
    const double seconds_per_byte_burst, seconds_per_byte;

    io_service_ref io_service;

    timer_type timer;
    /// Time at which next burst should be sent, considering the burst rate
    timer_type::time_point send_time_burst;
    /// Time at which next burst should be sent, considering the average rate
    timer_type::time_point send_time;
    /// If true, rate_bytes is never incremented and hence we never sleep
    bool hw_rate = false;
    /// If true, we're not handing more packets until we've slept
    bool must_sleep = false;
    /// Number of bytes sent since send_time and sent_time_burst were updated
    std::uint64_t rate_bytes = 0;

    // Local copies of the head/tail pointers from the owning stream,
    // accessible without a lock.
    std::size_t queue_head = 0, queue_tail = 0;
    /// Entry from which we are currently getting new packets
    std::size_t active = 0;
    /**
     * Packet generator for the active heap. It may be empty at any time, which
     * indicates that it should be initialised from the heap indicated by
     * @ref active.
     *
     * When non-empty, it must always have a next packet i.e. after
     * exhausting it, it must be cleared/changed.
     */
    boost::optional<packet_generator> gen;
    stream *owner = nullptr;

    /**
     * Update @ref send_time_burst and @ref send_time from @ref rate_bytes.
     *
     * @param now       Current time
     * @returns         Time at which next packet should be sent
     */
    timer_type::time_point update_send_times(timer_type::time_point now);
    /**
     * Update @ref send_time after a period of no work.
     *
     * This is called by @ref stream when it wakes up the stream.
     */
    void update_send_time_empty();

    /// Called by stream constructor to set itself as owner.
    void set_owner(stream *owner);

    virtual void wakeup() = 0;

    /**
     * Called after setting the owner. The default behaviour is to call
     * @ref request_wakeup, but it may be overridden if that is not desired.
     */
    virtual void start() { request_wakeup(); }

protected:
    struct transmit_packet
    {
        packet pkt;
        std::size_t size;
        bool last;          // if this is the last packet in the heap
        stream::queue_item *item;
    };

    stream *get_owner() const { return owner; }

    /**
     * Derived class calls to indicate that it will take care of rate limiting in hardware.
     *
     * This must be called from the constructor as it is not thread-safe. The
     * caller must only call this if the stream config enabled HW rate limiting.
     */
    void enable_hw_rate();

    packet_result get_packet(transmit_packet &data);

    /// Notify the base class that @a n heaps have finished transmission.
    void heaps_completed(std::size_t n);

    /**
     * Request @ref wakeup once the sleep time has been reached. This must
     * be called after @ref get_packet returns @c packet_result::SLEEP.
     */
    void sleep();

    /**
     * Request @ref wakeup when new packets become available (new relative
     * to the last call to @ref get_packet).
     */
    void request_wakeup();

    /// Schedule wakeup to be called immediately.
    void post_wakeup();

    writer(io_service_ref io_service, const stream_config &config);

public:
    virtual ~writer() = default;

    /// Retrieve the io_service used for processing the stream
    boost::asio::io_service &get_io_service() const { return *io_service; }

    /// Number of substreams
    virtual std::size_t get_num_substreams() const = 0;
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_STREAM_H
