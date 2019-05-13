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

#ifndef SPEAD2_SEND_STREAM_H
#define SPEAD2_SEND_STREAM_H

#include <functional>
#include <utility>
#include <vector>
#include <memory>
#include <chrono>
#include <mutex>
#include <iterator>
#include <type_traits>
#include <condition_variable>
#include <stdexcept>
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

class stream_config
{
public:
    static constexpr std::size_t default_max_packet_size = 1472;
    static constexpr std::size_t default_max_heaps = 4;
    static constexpr std::size_t default_burst_size = 65536;
    static constexpr double default_burst_rate_ratio = 1.05;

    void set_max_packet_size(std::size_t max_packet_size);
    std::size_t get_max_packet_size() const;
    void set_rate(double rate);
    double get_rate() const;
    void set_burst_size(std::size_t burst_size);
    std::size_t get_burst_size() const;
    void set_max_heaps(std::size_t max_heaps);
    std::size_t get_max_heaps() const;
    void set_burst_rate_ratio(double burst_rate_ratio);
    double get_burst_rate_ratio() const;

    /// Get product of rate and burst_rate_ratio
    double get_burst_rate() const;

    explicit stream_config(
        std::size_t max_packet_size = default_max_packet_size,
        double rate = 0.0,
        std::size_t burst_size = default_burst_size,
        std::size_t max_heaps = default_max_heaps,
        double burst_rate_ratio = default_burst_rate_ratio);

private:
    std::size_t max_packet_size = default_max_packet_size;
    double rate = 0.0;
    std::size_t burst_size = default_burst_size;
    std::size_t max_heaps = default_max_heaps;
    double burst_rate_ratio = default_burst_rate_ratio;
};

/**
 * Abstract base class for streams.
 */
class stream
{
private:
    io_service_ref io_service;

protected:
    typedef std::function<void(const boost::system::error_code &ec, item_pointer_t bytes_transferred)> completion_handler;

    explicit stream(io_service_ref io_service);

public:
    /// Retrieve the io_service used for processing the stream
    boost::asio::io_service &get_io_service() const { return *io_service; }

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
    virtual void set_cnt_sequence(item_pointer_t next, item_pointer_t step) = 0;

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
     * @retval  false  If the heap was immediately discarded
     * @retval  true   If the heap was enqueued
     */
    virtual bool async_send_heap(const heap &h, completion_handler handler, s_item_pointer_t cnt = -1) = 0;

    /**
     * Block until all enqueued heaps have been sent. This function is
     * thread-safe, but can be live-locked if more heaps are added while it is
     * running.
     */
    virtual void flush() = 0;

    virtual ~stream();
};

template<typename Derived>
class stream_impl;

/**
 * Base class for @ref stream_impl. It implements the parts that are
 * independent of the transport.
 */
class stream_impl_base : public stream
{
    template<typename Derived> friend class stream_impl;
private:
    enum class state_t
    {
        QUEUED,
        SENDING,
        SLEEPING,
        EMPTY
    };

    typedef boost::asio::basic_waitable_timer<std::chrono::high_resolution_clock> timer_type;

    struct queue_item
    {
        const heap &h;
        item_pointer_t cnt;
        completion_handler handler;
        item_pointer_t bytes_sent = 0;

        queue_item() = default;
        queue_item(const heap &h, item_pointer_t cnt, completion_handler &&handler) noexcept
            : h(std::move(h)), cnt(cnt), handler(std::move(handler))
        {
        }
    };

    typedef std::aligned_storage<sizeof(queue_item), alignof(queue_item)>::type queue_item_storage;

protected:
    struct transmit_packet
    {
        packet pkt;
        std::size_t size;
        bool last;          // if this is the last packet in the heap
        queue_item *item;
        boost::system::error_code result;
    };

    std::unique_ptr<transmit_packet[]> current_packets;
    std::size_t n_current_packets = 0;
    const std::size_t max_current_packets;

private:
    const stream_config config;
    const double seconds_per_byte_burst, seconds_per_byte;

    /**
     * Protects access to
     * - @ref queue_head and @ref queue_tail
     * - @ref state
     * - @ref next_cnt and @ref step_cnt
     */
    std::mutex queue_mutex;
    /**
     * Circular queue with config.max_heaps + 1 slots, which must never be full
     * (because that can't be distinguished from empty). Items from @ref
     * queue_head to @ref queue_tail are constructed in place, which the rest
     * is uninitialised raw storage.
     */
    std::unique_ptr<queue_item_storage[]> queue;
    /// Bounds of the queue
    std::size_t queue_head = 0, queue_tail = 0;
    /// Item holding data for the next packet to send
    std::size_t active = 0;   // queue slot for next packet to send
    state_t state = state_t::EMPTY;
    timer_type timer;
    /// Time at which next burst should be sent, considering the burst rate
    timer_type::time_point send_time_burst;
    /// Time at which next burst should be sent, considering the average rate
    timer_type::time_point send_time;
    /// Number of bytes sent since send_time and sent_time_burst were updated
    std::uint64_t rate_bytes = 0;
    /// Heap cnt for the next heap to send
    item_pointer_t next_cnt = 1;
    /// Increment to next_cnt after each heap
    item_pointer_t step_cnt = 1;
    /**
     * Packet generator for the active heap. It may be empty at any time, which
     * indicates that it should be initialised from the heap indicated by
     * @ref active.
     *
     * When non-empty, it must always have a next packet i.e. after
     * exhausting it, it must be cleared/changed.
     */
    boost::optional<packet_generator> gen;
    /// Signalled when transitioning to EMPTY state
    std::condition_variable heap_empty;

    /// Get next slot position in queue
    std::size_t next_queue_slot(std::size_t cur) const;

    /// Access an item from the queue
    queue_item *get_queue(std::size_t idx);

    /**
     * Advance @ref active to the next heap and clear @ref gen.
     * It does not need @ref queue_mutex.
     */
    void next_active();

    /**
     * Set the result of the first heap in the queue and remove it. This must be
     * called with @ref queue_mutex held.
     */
    void post_handler(boost::system::error_code result);

    /**
     * Whether a full burst has been transmitted, requiring some sleep time.
     * Does not require @ref queue_mutex.
     */
    bool must_sleep() const;

    /**
     * Apply per-packet transmission results to the queue. This must be called
     * with @ref queue_mutex held.
     */
    void process_results();

    /**
     * Update @ref send_time_burst and @ref send_time from @ref rate_bytes.
     * Does not require @ref queue_mutex.
     *
     * @param now       Current time
     * @returns         Time at which next packet should be sent
     */
    timer_type::time_point update_send_times(timer_type::time_point now);

    /// Update @ref send_time after a period in state @c EMPTY.
    void update_send_time_empty();

    /**
     * Populate @ref current_packets and @ref n_current_packets with packets to
     * send. This is called without @ref queue_mutex held. It takes a copy of
     * @ref queue_tail so that it does not need the lock to access the original.
     */
    void load_packets(std::size_t tail);

protected:
    stream_impl_base(io_service_ref io_service, const stream_config &config, std::size_t max_current_packets);
    virtual ~stream_impl_base() override;

public:
    virtual void set_cnt_sequence(item_pointer_t next, item_pointer_t step) override;

    /**
     * Block until all enqueued heaps have been sent. This function is
     * thread-safe, but can be live-locked if more heaps are added while it is
     * running.
     */
    virtual void flush() override;
};

/**
 * Stream that sends packets at a maximum rate. It also serialises heaps so
 * that only one heap is being sent at a time. Heaps are placed in a queue, and
 * if the queue becomes too deep heaps are discarded.
 *
 * The stream operates as a state machine, depending on which handlers are
 * pending:
 * - QUEUED: was previously empty, but async_send_heap posted a callback to @ref do_next
 * - SENDING: the derived class is in the process of sending packets
 * - SLEEPING: we are sleeping as a result of rate limiting
 * - EMPTY: there are no heaps and no pending callbacks.
 *
 * The derived class implements @c async_send_packets, which is responsible for
 * arranging transmission the @ref n_current_packets stored in
 * @ref current_packets. Once the packets are sent, it must cause
 * @ref packets_handler to be called (but not before returning). It may assume
 * that there is at least one packet to send.
 *
 * There are two mechanisms to protect shared data: the @ref do_next callback
 * is only ever scheduled once, so it does not need to lock data for which it
 * is the only user. Data also accessed by user-facing functions are protected
 * by a mutex.
 */
template<typename Derived>
class stream_impl : public stream_impl_base
{
private:
    /**
     * Advance the state machine. Whenever the state is not EMPTY, there is a
     * future call to this function expected (possibly indirectly via
     * @ref packets_handler).
     */
    void do_next()
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

protected:
    /**
     * Report on completed packets. Each element of @a items must have it's @c result
     * field updated.
     *
     * This function must not be called directly from async_send_packets, as this will
     * lead to a deadlock. It must be scheduled to be called later.
     */
    void packets_handler()
    {
        do_next();
    }

    using stream_impl_base::stream_impl_base;

public:
    virtual bool async_send_heap(const heap &h, completion_handler handler, s_item_pointer_t cnt = -1) override
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
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_STREAM_H
