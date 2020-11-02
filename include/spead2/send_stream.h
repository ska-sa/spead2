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
#include <boost/optional.hpp>
#include <boost/system/error_code.hpp>
#include <spead2/send_heap.h>
#include <spead2/send_packet.h>
#include <spead2/send_stream_config.h>
#include <spead2/send_writer.h>
#include <spead2/common_logging.h>
#include <spead2/common_defines.h>
#include <spead2/common_thread_pool.h>

namespace spead2
{

namespace send
{

/// Determines how to order packets when using @ref spead2::send::stream::async_send_heaps.
enum class group_mode
{
    /**
     * Interleave the packets of the heaps. One packet is sent from each heap
     * in turn (skipping those that have run out of packets).
     */
    ROUND_ROBIN
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

    heap_reference(const send::heap &heap, s_item_pointer_t cnt = -1, std::size_t substream_index = 0)
        : heap(heap), cnt(cnt), substream_index(substream_index)
    {
    }
};

static inline const heap &get_heap(const heap &h)
{
    return h;
}

static inline s_item_pointer_t get_heap_cnt(const heap &h)
{
    return -1;
}

static inline std::size_t get_heap_substream_index(const heap &h)
{
    return 0;
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

    // These fields are only relevant for the first item in a group
    item_pointer_t bytes_sent = 0;
    boost::system::error_code result;
    completion_handler handler;
    // Populated by flush(). A forward_list takes less space when not used than vector.
    std::forward_list<std::promise<void>> waiters;

    queue_item() = default;
    queue_item(const heap &h, item_pointer_t cnt, std::size_t substream_index,
               std::size_t group_end, std::size_t group_next,
               std::size_t max_packet_size)
        : gen(h, cnt, max_packet_size),
        substream_index(substream_index),
        group_end(group_end), group_next(group_next)
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

    typedef std::aligned_storage<sizeof(detail::queue_item), alignof(detail::queue_item)>::type queue_item_storage;

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
    detail::queue_item *get_queue(std::size_t idx);

    /// No-op version of @ref unwinder (see below)
    class null_unwinder
    {
    public:
        explicit null_unwinder(stream &s, std::size_t tail) {}
        void set_tail(std::size_t tail) {}
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
        // Only mode so far - when we add more we'll need to update this function.
        assert(mode == group_mode::ROUND_ROBIN);
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
            if (substream_index >= num_substreams)
            {
                unwind.abort();
                lock.unlock();
                log_warning("async_send_heap(s): dropping heap because substream index is out of range");
                get_io_service().post(std::bind(std::move(handler), boost::asio::error::invalid_argument, 0));
                return false;
            }
            item_pointer_t cnt_mask = (item_pointer_t(1) << h.get_flavour().get_heap_address_bits()) - 1;

            if (tail - head == queue_size)
            {
                unwind.abort();
                lock.unlock();
                log_warning("async_send_heap(s): dropping heap because queue is full");
                get_io_service().post(std::bind(std::move(handler), boost::asio::error::would_block, 0));
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
                get_io_service().post(std::bind(std::move(handler), boost::asio::error::invalid_argument, 0));
                return false;
            }

            // Construct in place. The group values are set for a singleton,
            // and repaired later if that's not the case.
            auto *cur = get_queue(tail);
            new (cur) detail::queue_item(
                h, cnt, substream_index, tail + 1, tail,
                max_packet_size);
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
                cur->group_next = (i + 1 == tail) ? orig_tail : i + 1;
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
            get_io_service().post([w_ptr]() {
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
     * If this function returns @c false, the heap was rejected without
     * being added to the queue. The handler is called as soon as possible
     * (from a thread running the io_service). If the heap was rejected due to
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
     * (from a thread running the io_service). If the heaps were rejected due to
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
            get_io_service().post(std::bind(std::move(handler), boost::asio::error::invalid_argument, 0));
            return false;
        }
        return async_send_heaps_impl<unwinder, Iterator>(first, last, std::move(handler), mode);
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

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_STREAM_H
