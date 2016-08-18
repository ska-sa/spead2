/* Copyright 2015 SKA South Africa
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
#include <queue>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <boost/asio.hpp>
#include <boost/asio/high_resolution_timer.hpp>
#include <boost/system/error_code.hpp>
#include <spead2/send_heap.h>
#include <spead2/send_packet.h>
#include <spead2/common_logging.h>
#include <spead2/common_defines.h>

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

    void set_max_packet_size(std::size_t max_packet_size);
    std::size_t get_max_packet_size() const;
    void set_rate(double rate);
    double get_rate() const;
    void set_burst_size(std::size_t burst_size);
    std::size_t get_burst_size() const;
    void set_max_heaps(std::size_t max_heaps);
    std::size_t get_max_heaps() const;

    explicit stream_config(
        std::size_t max_packet_size = default_max_packet_size,
        double rate = 0.0,
        std::size_t burst_size = default_burst_size,
        std::size_t max_heaps = default_max_heaps);

private:
    std::size_t max_packet_size = default_max_packet_size;
    double rate = 0.0;
    std::size_t burst_size = default_burst_size;
    std::size_t max_heaps = default_max_heaps;
};

/**
 * Abstract base class for streams.
 */
class stream
{
private:
    boost::asio::io_service &io_service;

protected:
    typedef std::function<void(const boost::system::error_code &ec, item_pointer_t bytes_transferred)> completion_handler;

    explicit stream(boost::asio::io_service &io_service);

public:
    /// Retrieve the io_service used for processing the stream
    boost::asio::io_service &get_io_service() const { return io_service; }

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

/**
 * Stream that sends packets at a maximum rate. It also serialises heaps so
 * that only one heap is being sent at a time. Heaps are placed in a queue, and if
 * the queue becomes too long heaps are discarded.
 */
template<typename Derived>
class stream_impl : public stream
{
private:
    typedef boost::asio::basic_waitable_timer<std::chrono::high_resolution_clock> timer_type;

    struct queue_item
    {
        const heap &h;
        item_pointer_t cnt;
        completion_handler handler;

        queue_item() = default;
        queue_item(const heap &h, item_pointer_t cnt, completion_handler &&handler)
            : h(std::move(h)), cnt(cnt), handler(std::move(handler))
        {
        }
    };

    const stream_config config;
    const double seconds_per_byte;

    /**
     * Protects access to @a queue. All other members are either const or are
     * only accessed only by completion handlers, and there is only ever one
     * scheduled at a time.
     */
    std::mutex queue_mutex;
    std::queue<queue_item> queue;
    timer_type timer;
    timer_type::time_point send_time;
    /// Number of bytes sent in the current heap
    item_pointer_t heap_bytes = 0;
    /// Number of bytes sent since last sleep
    std::size_t rate_bytes = 0;
    /// Heap cnt for the next heap to send
    item_pointer_t next_cnt = 1;
    /// Increment to next_cnt after each heap
    item_pointer_t step_cnt = 1;
    std::unique_ptr<packet_generator> gen; // TODO: make this inlinable
    /// Packet undergoing transmission by send_next_packet
    packet current_packet;
    /// Signalled whenever the last heap is popped from the queue
    std::condition_variable heap_empty;

    /**
     * Asynchronously send the next packet from the current heap
     * (or the next heap, if the current one is finished).
     *
     * @param ec Error from sending the previous packet. If set, the rest of the
     *           current heap is aborted.
     */
    void send_next_packet(boost::system::error_code ec = boost::system::error_code())
    {
        bool again;
        do
        {
            assert(gen);
            again = false;
            current_packet = gen->next_packet();
            if (ec || current_packet.buffers.empty())
            {
                // Reached the end of a heap. Pop the current one, and start the
                // next one if there is one.
                completion_handler handler;
                bool empty;

                gen.reset();
                std::unique_lock<std::mutex> lock(queue_mutex);
                handler = std::move(queue.front().handler);
                queue.pop();
                empty = queue.empty();
                if (!empty)
                    gen.reset(new packet_generator(
                        queue.front().h, queue.front().cnt, config.get_max_packet_size()));
                else
                    gen.reset();

                std::size_t old_heap_bytes = heap_bytes;
                heap_bytes = 0;
                if (empty)
                {
                    heap_empty.notify_all();
                    // Avoid hanging on to data indefinitely
                    current_packet = packet();
                }
                again = !empty;  // Start again on the next heap
                lock.unlock();

                /* At this point it is not safe to touch *this at all, because
                 * if the queue is empty, the destructor is free to complete
                 * and take the memory out from under us.
                 */
                handler(ec, old_heap_bytes);
            }
            else
            {
                static_cast<Derived *>(this)->async_send_packet(
                    current_packet,
                    [this] (const boost::system::error_code &ec, std::size_t bytes_transferred)
                    {
                        if (ec)
                        {
                            send_next_packet(ec);
                            return;
                        }
                        bool sleeping = false;
                        rate_bytes += bytes_transferred;
                        heap_bytes += bytes_transferred;
                        if (rate_bytes >= config.get_burst_size())
                        {
                            std::chrono::duration<double> wait(rate_bytes * seconds_per_byte);
                            send_time += std::chrono::duration_cast<timer_type::clock_type::duration>(wait);
                            rate_bytes = 0;
                            auto now = timer_type::clock_type::now();
                            if (now < send_time)
                            {
                                sleeping = true;
                                timer.expires_at(send_time);
                                timer.async_wait([this] (const boost::system::error_code &error)
                                {
                                    send_next_packet(error);
                                });
                            }
                            // If we're behind schedule, we still keep send_time in the past,
                            // which will help with catching up if we oversleep
                        }
                        if (!sleeping)
                            send_next_packet();
                    });
            }
        } while (again);
    }

public:
    stream_impl(
        boost::asio::io_service &io_service,
        const stream_config &config = stream_config()) :
            stream(io_service),
            config(config),
            seconds_per_byte(config.get_rate() > 0.0 ? 1.0 / config.get_rate() : 0.0),
            timer(io_service)
    {
    }

    virtual void set_cnt_sequence(item_pointer_t next, item_pointer_t step) override
    {
        if (step == 0)
            throw std::invalid_argument("step cannot be 0");
        std::unique_lock<std::mutex> lock(queue_mutex);
        next_cnt = next;
        step_cnt = step;
    }

    virtual bool async_send_heap(const heap &h, completion_handler handler, s_item_pointer_t cnt = -1) override
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (queue.size() >= config.get_max_heaps())
        {
            log_warning("async_send_heap: dropping heap because queue is full");
            get_io_service().dispatch(std::bind(handler, boost::asio::error::would_block, 0));
            return false;
        }
        bool empty = queue.empty();
        item_pointer_t ucnt; // unsigned, so that copying next_cnt cannot overflow
        if (cnt < 0)
        {
            ucnt = next_cnt;
            next_cnt += step_cnt;
        }
        else
            ucnt = cnt;
        queue.emplace(h, ucnt, std::move(handler));
        if (empty)
        {
            assert(!gen);
            gen.reset(new packet_generator(queue.front().h, queue.front().cnt, config.get_max_packet_size()));
        }
        lock.unlock();

        /* If it is not empty, the new heap will be started as a continuation
         * of the previous one.
         */
        if (empty)
        {
            send_time = timer_type::clock_type::now();
            get_io_service().dispatch([this] { send_next_packet(); });
        }
        return true;
    }

    /**
     * Block until all enqueued heaps have been sent. This function is
     * thread-safe, but can be live-locked if more heaps are added while it is
     * running.
     */
    virtual void flush() override
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        while (!queue.empty())
        {
            heap_empty.wait(lock);
        }
    }

    ~stream_impl()
    {
        // TODO: add a stop member to abort transmission and use that instead
        flush();
    }
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_STREAM_H
