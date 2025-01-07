/* Copyright 2015, 2017, 2019-2020, 2023-2025 National Research Foundation (SARAO)
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
#include <spead2/send_packet.h>
#include <spead2/send_stream_config.h>
#include <spead2/send_writer.h>
#include <spead2/common_logging.h>
#include <spead2/common_defines.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_storage.h>

namespace spead2::send
{

/// Determines how to order packets when using @ref spead2::send::stream::async_send_heaps.
enum class group_mode
{
    /**
     * Interleave the packets of the heaps. One packet is sent from each heap
     * in turn (skipping those that have run out of packets).
     */
    ROUND_ROBIN,
    /**
     * Send the heaps serially, as if they were added one at a time.
     */
    SERIAL
};

/**
 * Associate a heap with metadata needed to transmit it.
 *
 * It holds a reference to the original heap.
 */
struct heap_reference
{
    const send::heap &heap;
    s_item_pointer_t cnt;
    std::size_t substream_index;
    double rate;

    heap_reference(
        const send::heap &heap,
        s_item_pointer_t cnt = -1,
        std::size_t substream_index = 0,
        double rate = -1.0  // negative means to use the stream's rate
    )
        : heap(heap), cnt(cnt), substream_index(substream_index), rate(rate)
    {
    }
};

static inline const heap &get_heap(const heap &h)
{
    return h;
}

static inline s_item_pointer_t get_heap_cnt(const heap &)
{
    return -1;
}

static inline std::size_t get_heap_substream_index(const heap &)
{
    return 0;
}

static inline double get_heap_rate(const heap &)
{
    return -1.0;
}

static inline const heap &get_heap(const heap_reference &ref)
{
    return ref.heap;
}

static inline s_item_pointer_t get_heap_cnt(const heap_reference &ref)
{
    return ref.cnt;
}

static inline std::size_t get_heap_substream_index(const heap_reference &ref)
{
    return ref.substream_index;
}

static inline double get_heap_rate(const heap_reference &ref)
{
    return ref.rate;
}

namespace detail
{

/* Entry in a stream's queue. It is logically an inner class of stream, but
 * C++ doesn't allow forward references to inner classes so it is split out.
 */
struct queue_item
{
    typedef std::function<void(const boost::system::error_code &ec, item_pointer_t bytes_transferred)> completion_handler;

    packet_generator gen;
    const std::size_t substream_index;
    // Queue index (non-masked) one past the end of a group
    std::size_t group_end;
    /* Next queue index (non-masked) to send a packet from after this one.
     * This gets updated lazily when that one is exhausted, so that it
     * requires amortised constant time to find the next non-exhausted
     * item in the group.
     */
    std::size_t group_next;
    // Type of group
    group_mode mode;
    precise_time::correction_type wait_per_byte;

    // These fields are only relevant for the first item in a group
    item_pointer_t bytes_sent = 0;
    boost::system::error_code result;
    completion_handler handler;
    // Populated by flush(). A forward_list takes less space when not used than vector.
    std::forward_list<std::promise<void>> waiters;

    queue_item(const heap &h, item_pointer_t cnt, std::size_t substream_index,
               std::size_t group_end, std::size_t group_next, group_mode mode,
               std::size_t max_packet_size,
               precise_time::correction_type wait_per_byte)
        : gen(h, cnt, max_packet_size),
        substream_index(substream_index),
        group_end(group_end), group_next(group_next), mode(mode),
        wait_per_byte(wait_per_byte)
    {
    }
};

} // namespace detail

/**
 * Stream for sending heaps, potentially to multiple destinations.
 */
class stream
{
public:
    typedef detail::queue_item::completion_handler completion_handler;

private:
    friend class writer;

    typedef spead2::detail::storage<detail::queue_item> queue_item_storage;

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
    /// Maximum packet size, copied from the stream config
    const std::size_t max_packet_size;
    /// Duration corresponding to one byte at the default rate
    const detail::precise_time::correction_type default_wait_per_byte;
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
    bool need_wakeup = false;

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

    /// Access the storage for a queue item (takes care of masking the index)
    queue_item_storage &get_queue_storage(std::size_t idx);

    /// Access a (valid) item from the queue (takes care of masking the index)
    detail::queue_item *get_queue(std::size_t idx);

    /// No-op version of @ref unwinder (see below)
    class null_unwinder
    {
    public:
        explicit null_unwinder(stream &, std::size_t) {}
        void set_tail(std::size_t) {}
        void abort() {}
        void commit() {}
    };

    /**
     * Unwinds partial application of @ref async_send_heaps. Specifically, it
     * destroys constructed @ref detail::queue_item entries in the queue.
     */
    class unwinder
    {
    private:
        stream &s;
        std::size_t orig_tail;    ///< Tail when the unwinder was constructed
        std::size_t tail;         ///< Current tail

    public:
        explicit unwinder(stream &s, std::size_t tail);
        ~unwinder() { abort(); }
        /**
         * Update the current tail. All entries in [orig_tail, tail) will be
         * destroyed by the destructor.
         */
        void set_tail(std::size_t tail);
        /// Perform the unwinding immediately.
        void abort();
        /// Prevent the unwinding from happening, because async_send_heaps was successful.
        void commit();
    };

    /// Common implementation for @ref async_send_heap and @ref async_send_heaps
    template<typename Unwinder, typename Iterator>
    bool async_send_heaps_impl(Iterator first, Iterator last,
                               completion_handler &&handler, group_mode mode)
    {
        // Need a slot to put the handler in; caller must handle the exception case
        assert(first != last);
        // Only modes so far - when we add more we'll need to update this function.
        assert(mode == group_mode::ROUND_ROBIN || mode == group_mode::SERIAL);
        std::unique_lock<std::mutex> lock(tail_mutex);
        std::size_t tail = queue_tail.load(std::memory_order_relaxed);
        std::size_t orig_tail = tail;
        std::size_t head = queue_head.load(std::memory_order_acquire);
        std::size_t next_cnt = this->next_cnt;
        Unwinder unwind(*this, tail);

        for (Iterator it = first; it != last; ++it)
        {
            const heap &h = get_heap(*it);
            s_item_pointer_t cnt = get_heap_cnt(*it);
            std::size_t substream_index = get_heap_substream_index(*it);
            double rate = get_heap_rate(*it);
            detail::precise_time::correction_type wait_per_byte;
            if (rate == 0.0)
            {
                wait_per_byte = wait_per_byte.zero();
            }
            else if (rate > 0.0)
            {
                wait_per_byte = std::chrono::duration<double>(1.0 / rate);
            }
            else
            {
                wait_per_byte = default_wait_per_byte;
            }
            if (substream_index >= num_substreams)
            {
                unwind.abort();
                lock.unlock();
                log_warning("async_send_heap(s): dropping heap because substream index is out of range");
                boost::asio::post(get_io_context(), std::bind(std::move(handler), boost::asio::error::invalid_argument, 0));
                return false;
            }
            item_pointer_t cnt_mask = (item_pointer_t(1) << h.get_flavour().get_heap_address_bits()) - 1;

            if (tail - head == queue_size)
            {
                unwind.abort();
                lock.unlock();
                log_warning("async_send_heap(s): dropping heap because queue is full");
                boost::asio::post(get_io_context(), std::bind(std::move(handler), boost::asio::error::would_block, 0));
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
                log_warning("async_send_heap(s): dropping heap because cnt is out of range");
                boost::asio::post(get_io_context(), std::bind(std::move(handler), boost::asio::error::invalid_argument, 0));
                return false;
            }

            // Construct in place. The group values are set for a singleton,
            // and repaired later if that's not the case.
            get_queue_storage(tail).construct(
                h, cnt, substream_index, tail + 1, tail, mode,
                max_packet_size,
                wait_per_byte);
            tail++;
            unwind.set_tail(tail);
        }

        // We've successfully added all the heaps, so start commiting the changes
        get_queue(orig_tail)->handler = std::move(handler);
        if (tail != orig_tail + 1)
        {
            for (std::size_t i = orig_tail; i != tail; i++)
            {
                auto *cur = get_queue(i);
                cur->group_end = tail;
                if (i + 1 == tail)
                {
                    /* In ROUND_ROBIN mode, cycle back around to the start.
                     * In SERIAL mode, mark the last heap as terminal.
                     */
                    cur->group_next = (mode == group_mode::ROUND_ROBIN) ? orig_tail : i;
                }
                else
                    cur->group_next = i + 1;
            }
        }
        this->next_cnt = next_cnt;
        unwind.commit();

        bool wakeup = need_wakeup;
        need_wakeup = false;
        queue_tail.store(tail, std::memory_order_release);
        lock.unlock();
        if (wakeup)
        {
            writer *w_ptr = w.get();
            boost::asio::post(get_io_context(), [w_ptr]() {
                w_ptr->update_send_time_empty();
                w_ptr->wakeup();
            });
        }
        return true;
    }

protected:
    writer &get_writer() { return *w; }
    const writer &get_writer() const { return *w; }

    explicit stream(std::unique_ptr<writer> &&w);

public:
    /// Retrieve the io_context used for processing the stream
    boost::asio::io_context &get_io_context() const;
    /// Retrieve the io_context used for processing the stream (deprecated)
    [[deprecated("use get_io_context")]]
    boost::asio::io_context &get_io_service() const;

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
     * If this function returns @c false, the heap was rejected without
     * being added to the queue. The handler is called as soon as possible
     * (from a thread running the io_context). If the heap was rejected due to
     * lack of space, the error code is @c boost::asio::error::would_block.
     *
     * By default the heap cnt is chosen automatically (see @ref set_cnt_sequence).
     * An explicit value can instead be chosen by passing a non-negative value
     * for @a cnt. When doing this, it is entirely the responsibility of the
     * user to avoid collisions, both with other explicit values and with the
     * automatic counter. This feature is useful when multiple senders
     * contribute to a single stream and must keep their heap cnts disjoint,
     * which the automatic assignment would not do.
     *
     * The transmission rate may be overridden using the optional @a rate
     * parameter. If it is negative, the stream's rate applies, if it is zero
     * there is no rate limiting, and if it is positive it specifies the rate
     * in bytes per second.
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
                         std::size_t substream_index = 0,
                         double rate = -1.0);

    /**
     * Send @a h asynchronously, with an arbitrary completion token. This
     * overload is not used if the completion token is convertible to
     * @ref completion_handler.
     *
     * Refer to the other overload for details. The boolean return of the other
     * overload is absent. You will need to retrieve the asynchronous result
     * and check for @c boost::asio::error::would_block to determine if the
     * heaps were rejected due to lack of buffer space.
     */
    template<typename CompletionToken>
    auto async_send_heap(const heap &h, CompletionToken &&token,
                         s_item_pointer_t cnt = -1,
                         std::enable_if_t<
                            !std::is_convertible_v<CompletionToken, completion_handler>,
                            std::size_t
                         > substream_index = 0,
                         double rate = -1.0)
    {
        auto init = [this, &h, cnt, substream_index, rate](auto handler)
        {
            // Explicit this-> is to work around bogus warning from clang
            this->async_send_heap(h, std::move(handler), cnt, substream_index, rate);
        };
        return boost::asio::async_initiate<
            CompletionToken, void(const boost::system::error_code &, item_pointer_t)
        >(init, token);
    }

    /**
     * Send a group of heaps asynchronously, with @a handler called on
     * completion. The caller must ensure that the @ref heap objects
     * (as well as any memory they point to) remain valid until @a handler is
     * called.
     *
     * If this function returns @c true, then the heaps have been added to the
     * queue. The completion handlers for such heaps are guaranteed to be
     * called in order. Note that there is no individual per-heap feedback;
     * the callback is called once to give the result of the entire group.
     *
     * If this function returns @c false, the heaps were rejected without
     * being added to the queue. The handler is called as soon as possible
     * (from a thread running the io_context). If the heaps were rejected due to
     * lack of space, the error code is @c boost::asio::error::would_block.
     * It is an error to send an empty list of heaps.
     *
     * Note that either all the heaps will be queued, or none will; in
     * particular, there needs to be enough space in the queue for them all.
     *
     * The heaps are specified by a range of input iterators. Typically they
     * will be of type @ref heap_reference, but other types can be used by
     * overloading @c get_heap, @c get_heap_cnt and @c
     * get_heap_substream_index for the value type of the iterator. Refer to
     * @ref async_send_heap for an explanation of the @a cnt and @a
     * substream_index parameters.
     *
     * The @ref heap_reference objects can be safely deleted once this
     * function returns; it is sufficient for the @ref heap objects (and the
     * data they reference) to persist.
     *
     * @retval  false  If the heaps were immediately discarded
     * @retval  true   If the heaps were enqueued
     */
    template<typename Iterator>
    bool async_send_heaps(Iterator first, Iterator last,
                          completion_handler handler, group_mode mode)
    {
        if (first == last)
        {
            log_warning("Empty heap group");
            boost::asio::post(get_io_context(), std::bind(std::move(handler), boost::asio::error::invalid_argument, 0));
            return false;
        }
        return async_send_heaps_impl<unwinder, Iterator>(first, last, std::move(handler), mode);
    }

    /**
     * Send a group of heaps asynchronously, with an arbitrary completion
     * token (e.g., @c boost::asio::use_future). This overload is not used
     * if the completion token is convertible to @ref completion_handler.
     *
     * Refer to the other overload for details. There are a few differences:
     *
     * - The boolean return of the other overload is absent. You will need to
     *   retrieve the asynchronous result and check for @c
     *   boost::asio::error::would_block to determine if the heaps were
     *   rejected due to lack of buffer space.
     * - Depending on the completion token, the iterators might not be used
     *   immediately. Using @c boost::asio::use_future causes them to be used
     *   immediately, but @c boost::asio::deferred or @c
     *   boost::asio::use_awaitable does not (they are only used when
     *   awaiting the result). If they are not used immediately, the caller
     *   must keep them valid (as well as the data they reference) until they
     *   are used.
     */
    template<typename Iterator, typename CompletionToken>
    auto async_send_heaps(Iterator first, Iterator last,
                          CompletionToken &&token,
                          std::enable_if_t<
                              !std::is_convertible_v<CompletionToken, completion_handler>,
                              group_mode
                          > mode)
    {
        auto init = [this, first, last, mode](auto handler)
        {
            // Explicit this-> is to work around bogus warning from clang
            this->async_send_heaps(first, last, std::move(handler), mode);
        };
        return boost::asio::async_initiate<
            CompletionToken, void(const boost::system::error_code &, item_pointer_t)
        >(init, token);
    }

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

extern template bool stream::async_send_heaps_impl<stream::null_unwinder, heap_reference *>(
    heap_reference *first, heap_reference *last,
    completion_handler &&handler, group_mode mode);

} // namespace spead2::send

#endif // SPEAD2_SEND_STREAM_H
