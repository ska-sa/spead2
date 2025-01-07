/* Copyright 2020, 2023-2025 National Research Foundation (SARAO)
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

#ifndef SPEAD2_SEND_WRITER_H
#define SPEAD2_SEND_WRITER_H

#include <chrono>
#include <cstdint>
#include <boost/asio.hpp>
#include <boost/asio/high_resolution_timer.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/send_packet.h>
#include <spead2/send_stream_config.h>

namespace spead2::send
{

namespace detail { struct queue_item; }
class stream;

namespace detail
{

/**
 * A time with sub-nanosecond precision. See dev-send-rate-limit.rst for
 * an explanation.
 */
class precise_time
{
public:
    using timer_type = boost::asio::basic_waitable_timer<std::chrono::high_resolution_clock>;
    using coarse_type = timer_type::time_point;
    using correction_type = std::chrono::duration<double, timer_type::time_point::period>;
private:
    coarse_type coarse{};
    correction_type correction{0};

    void normalize();  // ensure correction is in [0, 1)
public:
    precise_time() = default;
    precise_time(const coarse_type &coarse);
    precise_time &operator+=(const correction_type &delta);
    bool operator<(const precise_time &other) const;

    const coarse_type &get_coarse() const { return coarse; }
};

} // namespace detail

/**
 * Back-end for a @ref stream. A writer is responsible for retrieving packets
 * and transmitting them, and calling the user-provided handlers when heaps are
 * completed.
 *
 * Each stream class will need to implement a subclass of @ref writer. At a
 * minimum, it will need to implement @ref wakeup and @ref get_num_substreams.
 *
 * A writer is intended to run on an io_context. It is *not* thread-safe, so
 * the subclass must ensure that only one handler runs at a time.
 *
 * The @ref wakeup handler should use @ref get_packet to try to retrieve
 * packet(s) and send them, and should ensure that @ref groups_completed is
 * called after transmitting final packets of groups. It is also responsible
 * for updating queue_item::bytes_sent and
 * queue_item::result. Depending on the last result of @ref
 * get_packet, it should arrange for itself to be rerun by calling either
 * @ref sleep, @ref request_wakeup or @ref post_wakeup.
 *
 * It is not safe to call @ref wakeup (or @ref post_wakeup) immediately after
 * construction, because the associated stream is not yet known. If there is
 * initialisation that has the potential to interact with the stream
 * (including by posting a callback, which the io_context might immediately run
 * on another thread), it should be done by overriding @ref start.
 */
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

    using precise_time = detail::precise_time;
    using timer_type = precise_time::timer_type;

    const stream_config config;    // TODO: probably doesn't need the whole thing

    io_context_ref io_context;

    /// Timer for sleeping for rate limiting
    timer_type timer;
    /// Time at which next burst should be sent, considering the burst rate
    precise_time send_time_burst;
    /// Time at which next burst should be sent, considering the average rate
    precise_time send_time;
    /// If true, rate_bytes is never incremented and hence we never sleep
    bool hw_rate = false;
    /// If true, we're not handling more packets until we've slept
    bool must_sleep = false;
    /// Number of bytes sent since send_time and sent_time_burst were updated
    std::uint64_t rate_bytes = 0;
    /// Amount to add to send_time on next update
    precise_time::correction_type rate_wait{0};

    // Local copies of the head/tail pointers from the owning stream,
    // accessible without a lock.
    std::size_t queue_head = 0, queue_tail = 0;
    /// Entry from which we are currently getting new packets
    std::size_t active = 0;
    /// Start of group containing active
    std::size_t active_start = 0;
    /**
     * The stream with which we're associated. This is filled in by @ref
     * set_owner shortly after construction.
     */
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

    /**
     * Implementation of the transport. See the class documentation for details.
     */
    virtual void wakeup() = 0;

    /**
     * Called after setting the owner. The default behaviour is to call
     * @ref request_wakeup, but it may be overridden if that is not desired.
     */
    virtual void start() { request_wakeup(); }

protected:
    struct transmit_packet
    {
        std::vector<boost::asio::const_buffer> buffers;
        std::size_t size;
        std::size_t substream_index;
        bool last;          // if this is the last packet in the group
        detail::queue_item *item;
    };

    stream *get_owner() const { return owner; }

    /**
     * Derived class calls to indicate that it will take care of rate limiting in hardware.
     *
     * This must be called from the constructor as it is not thread-safe. The
     * caller must only call this if the stream config enabled HW rate limiting.
     */
    void enable_hw_rate();

    /**
     * Retrieve a packet from the stream.
     *
     * If successful, the packet information is written into @a data.
     */
    packet_result get_packet(transmit_packet &data, std::uint8_t *scratch);

    /// Notify the base class that @a n groups have finished transmission.
    void groups_completed(std::size_t n);

    /**
     * Request @ref wakeup once the sleep time has been reached. This must
     * be called after @ref get_packet returns @c packet_result::SLEEP.
     */
    void sleep();

    /**
     * Request @ref wakeup when new packets become available (new relative
     * to the last call to @ref get_packet).
     *
     * Note: if this is called and there are no heaps whose callbacks are
     * still outstanding, this writer may be immediately destroyed by another
     * thread.
     */
    void request_wakeup();

    /// Schedule wakeup to be called immediately.
    void post_wakeup();

    writer(io_context_ref io_context, const stream_config &config);

public:
    virtual ~writer() = default;

    /// Retrieve the io_context used for processing the stream
    boost::asio::io_context &get_io_context() const { return *io_context; }
    /// Retrieve the io_context used for processing the stream (deprecated)
    [[deprecated("use get_io_context")]]
    boost::asio::io_context &get_io_service() const { return *io_context; }

    /// Number of substreams
    virtual std::size_t get_num_substreams() const = 0;
};

} // namespace spead2::send

#endif // SPEAD2_SEND_WRITER_H
