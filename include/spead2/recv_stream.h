/* Copyright 2015, 2017, 2018 SKA South Africa
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

#ifndef SPEAD2_RECV_STREAM_H
#define SPEAD2_RECV_STREAM_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <functional>
#include <future>
#include <mutex>
#include <atomic>
#include <type_traits>
#include <boost/asio.hpp>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_reader.h>
#include <spead2/common_memory_pool.h>
#include <spead2/common_bind.h>

namespace spead2
{

class thread_pool;
class io_service_ref;

namespace recv
{

struct packet_header;

/**
 * Statistics about a stream. Not all fields are relevant for all stream types.
 */
struct stream_stats
{
    /// Total number of heaps passed to @ref stream_base::heap_ready
    std::uint64_t heaps = 0;
    /**
     * Number of incomplete heaps that were evicted from the buffer to make
     * room for new data.
     */
    std::uint64_t incomplete_heaps_evicted = 0;
    /**
     * Number of incomplete heaps that were emitted by @ref stream::flush.
     * These are typically heaps that were in-flight when the stream stopped.
     */
    std::uint64_t incomplete_heaps_flushed = 0;
    /// Number of packets received
    std::uint64_t packets = 0;
    /// Number of batches of packets.
    std::uint64_t batches = 0;
    /**
     * Number of times a worker thread was blocked because the ringbuffer was
     * full. Only applicable to @ref ring_stream.
     */
    std::uint64_t worker_blocked = 0;
    /**
     * Maximum number of packets received as a unit. This is only applicable
     * to readers that support fetching a batch of packets from the source.
     */
    std::size_t max_batch = 0;

    /**
     * Number of heaps that were entirely contained in one packet.
     */
    std::uint64_t single_packet_heaps = 0;

    /// Total number of hash table probes.
    std::uint64_t search_dist = 0;

    stream_stats operator+(const stream_stats &other) const;
    stream_stats &operator+=(const stream_stats &other);
};

/**
 * Encapsulation of a SPEAD stream. Packets are fed in through @ref add_packet.
 * The base class does nothing with heaps; subclasses will typically override
 * @ref heap_ready and @ref stop_received to do further processing.
 *
 * A collection of partial heaps is kept. Heaps are removed from this collection
 * and passed to @ref heap_ready when
 * - They are known to be complete (a heap length header is present and all the
 *   corresponding payload has been received); or
 * - Too many heaps are live: the one with the lowest ID is aged out, even if
 *   incomplete
 * - The stream is stopped
 *
 * This class is @em not thread-safe. Almost all use cases (possibly excluding
 * testing) will derive from @ref stream.
 *
 * @internal
 *
 * The live heaps are stored in a circular queue (this has fewer pointer
 * indirections than @c std::deque).
 * When a heap is removed from the circular queue, the queue is not shifted
 * up. Instead, a hole is left. The queue thus only needs a head and not a
 * tail. When adding a new heap, any heap stored in the head position is
 * evicted. This means that heaps may be evicted before it is strictly
 * necessary from the point of view of available storage, but this prevents
 * heaps with lost packets from hanging around forever.
 *
 * A hash table is used to accelerate finding the live heap matching the
 * incoming packet. Some care is needed, because some hash table
 * implementations just take the lower bits of a number to map it to a bucket,
 * and some SPEAD streams increment cnts by a power of two, which can easily
 * lead to all heaps in the same bucket. So rather than using
 * std::unordered_map, we use a custom hash table implementation (with a
 * fixed number of buckets).
 */
class stream_base
{
private:
    struct queue_entry
    {
        queue_entry *next;   // Hash table chain
        live_heap heap;
        /* TODO: pad to a multiple of 16 bytes, so that there is a
         * good chance of next and heap.cnt being in the same cache line.
         */
    };

    typedef typename std::aligned_storage<sizeof(queue_entry), alignof(queue_entry)>::type storage_type;
    /**
     * Circular queue for heaps.
     *
     * A particular heap is in a constructed state iff the next pointer is
     * not INVALID_ENTRY.
     */
    std::unique_ptr<storage_type[]> queue_storage;
    /// Number of entries in @ref buckets
    std::size_t bucket_count;
    /// Right shift to map 64-bit unsigned to a bucket index
    int bucket_shift;
    /// Pointer to the first heap in each bucket, or NULL
    std::unique_ptr<queue_entry *[]> buckets;
    /// Position of the most recently added heap
    std::size_t head;

    /// Maximum number of live heaps permitted.
    std::size_t max_heaps;
    /// Protocol bugs to be compatible with
    bug_compat_mask bug_compat;

    /// Function used to copy heap payloads
    std::atomic<memcpy_function> memcpy{std::memcpy};
    /// Whether to stop when a stream control stop item is received
    std::atomic<bool> stop_on_stop_item{true};

    /// Mutex protecting @ref allocator
    std::mutex allocator_mutex;
    /**
     * Memory allocator used by heaps.
     *
     * This is protected by allocator_mutex. C++11 mandates free @c atomic_load
     * and @c atomic_store on @c shared_ptr, but GCC 4.8 doesn't implement it.
     * Also, std::atomic<std::shared_ptr<T>> causes undefined symbol errors, and
     * is illegal because shared_ptr is not a POD type.
     */
    std::shared_ptr<memory_allocator> allocator;

    /// @ref stop_received has been called, either externally or by stream control
    std::atomic<bool> stopped{false};

    /// Compute bucket number for a heap cnt
    std::size_t get_bucket(s_item_pointer_t heap_cnt) const;

    /// Get an entry from @ref queue_storage with the right type
    queue_entry *cast(std::size_t index);

    /**
     * Unlink an entry from the hash table.
     *
     * Note: this must be called before moving away the underlying heap,
     * as it depends on accessing the heap cnt to identify the bucket.
     */
    void unlink_entry(queue_entry *entry);

    /**
     * Callback called when a heap is being ejected from the live list.
     * The heap might or might not be complete.
     */
    virtual void heap_ready(live_heap &&) {}

protected:
    mutable std::mutex stats_mutex;
    stream_stats stats;

public:
    /**
     * State for a batch of calls to @ref add_packet. The strand must be held for
     * its entire lifetime. It holds copies of data that otherwise require
     * atomic access in the owning class.
     */
    struct add_packet_state
    {
        stream_base &owner;

        // Copied from the stream, but unencumbered by locks/atomics
        memcpy_function memcpy;
        std::shared_ptr<memory_allocator> allocator;
        bool stop_on_stop_item;
        // Updates to the statistics
        std::uint64_t packets = 0;
        std::uint64_t complete_heaps = 0;
        std::uint64_t incomplete_heaps_evicted = 0;
        std::uint64_t single_packet_heaps = 0;
        std::uint64_t search_dist = 0;

        explicit add_packet_state(stream_base &owner);
        ~add_packet_state();
    };

    static constexpr std::size_t default_max_heaps = 4;

    /**
     * Constructor.
     *
     * @param bug_compat   Protocol bugs to have compatibility with
     * @param max_heaps    Maximum number of live (in-flight) heaps held in the stream
     */
    explicit stream_base(bug_compat_mask bug_compat = 0, std::size_t max_heaps = default_max_heaps);
    virtual ~stream_base();

    /**
     * Set a pool to use for allocating heap memory.
     *
     * @deprecated Use @ref spead2::recv::stream_base::set_memory_allocator instead.
     */
    void set_memory_pool(std::shared_ptr<memory_pool> pool);

    /**
     * Set an allocator to use for allocating heap memory.
     */
    void set_memory_allocator(std::shared_ptr<memory_allocator> allocator);

    /// Set an alternative memcpy function for copying heap payload
    void set_memcpy(memcpy_function memcpy);

    /// Set builtin memcpy function to use for copying payload
    void set_memcpy(memcpy_function_id id);

    /// Set whether to stop the stream when a stop item is received
    void set_stop_on_stop_item(bool stop);

    /// Get whether to stop the stream when a stop item is received
    bool get_stop_on_stop_item() const;

    /**
     * Add a packet that was received, and which has been examined by @a
     * decode_packet, and returns @c true if it is consumed. Even though @a
     * decode_packet does some basic sanity-checking, it may still be rejected
     * by @ref live_heap::add_packet e.g., because it is a duplicate.
     *
     * It is an error to call this after the stream has been stopped.
     */
    bool add_packet(add_packet_state &state, const packet_header &packet);

    /**
     * Shut down the stream. This calls @ref flush.  Subclasses may override
     * this to achieve additional effects, but must chain to the base
     * implementation.
     *
     * It is undefined what happens if @ref add_packet is called after a stream
     * is stopped.
     */
    virtual void stop_received();

    bool is_stopped() const { return stopped.load(); }

    bug_compat_mask get_bug_compat() const { return bug_compat; }

    /// Flush the collection of live heaps, passing them to @ref heap_ready.
    void flush();
};

/**
 * Stream that is fed by subclasses of @ref reader. Unless otherwise specified,
 * methods in @ref stream_base may only be called while holding the strand
 * contained in this class. The public interface functions must be called
 * from outside the strand (and outside the threads associated with the
 * io_service), but are not thread-safe relative to each other.
 *
 * This class is thread-safe. This is achieved mostly by having operations run
 * as completion handlers on a strand. The exception is @ref stop, which uses a
 * @c once to ensure that only the first call actually runs.
 */
class stream : protected stream_base
{
private:
    friend class reader;

    /// Holder that just ensures that the thread pool doesn't vanish
    std::shared_ptr<thread_pool> thread_pool_holder;

    /**
     * Serialization of access.
     */
    boost::asio::io_service::strand strand;
    /**
     * Readers providing the stream data.
     */
    std::vector<std::unique_ptr<reader> > readers;

    /// Ensure that @ref stop is only run once
    std::once_flag stop_once;

    template<typename T, typename... Args>
    void emplace_reader_callback(Args&&... args)
    {
        if (!is_stopped())
        {
            // Guarantee space before constructing the reader
            readers.emplace_back(nullptr);
            readers.pop_back();
            std::unique_ptr<reader> ptr(reader_factory<T>::make_reader(*this, std::forward<Args>(args)...));
            if (ptr->lossy())
                lossy = true;
            readers.push_back(std::move(ptr));
        }
    }

    /* Prevent moving (copying is already impossible). Moving is not safe
     * because readers refer back to *this (it could potentially be added if
     * there is a good reason for it, but it would require adding a new
     * function to the reader interface.
     */
    stream(stream_base &&) = delete;
    stream &operator=(stream_base &&) = delete;

protected:
    virtual void stop_received() override;

    /**
     * Schedule execution of the function object @a callback through the @c
     * io_service using the strand, and block until it completes. If the
     * function throws an exception, it is rethrown in this thread.
     */
    template<typename F>
    typename std::result_of<F()>::type run_in_strand(F &&func)
    {
        typedef typename std::result_of<F()>::type return_type;
        std::packaged_task<return_type()> task(std::forward<F>(func));
        auto future = task.get_future();
        get_strand().dispatch([&task]
        {
            /* This is subtle: task lives on the run_in_strand stack frame, so
             * we have to be very careful not to touch it after that function
             * exits. Calling task() directly can continue to touch task even
             * after it has unblocked the future. But the move constructor for
             * packaged_task will take over the shared state for the future.
             */
            std::packaged_task<return_type()> my_task(std::move(task));
            my_task();
        });
        return future.get();
    }

    /// True if any lossy reader has been added (only access with strand held)
    bool lossy;

    /// Actual implementation of @ref stop
    void stop_impl();

public:
    using stream_base::get_bug_compat;
    using stream_base::default_max_heaps;
    using stream_base::set_memory_pool;
    using stream_base::set_memory_allocator;
    using stream_base::set_memcpy;
    using stream_base::set_stop_on_stop_item;
    using stream_base::get_stop_on_stop_item;

    boost::asio::io_service::strand &get_strand() { return strand; }

    explicit stream(io_service_ref io_service, bug_compat_mask bug_compat = 0, std::size_t max_heaps = default_max_heaps);
    virtual ~stream() override;

    /**
     * Add a new reader by passing its constructor arguments, excluding
     * the initial @a stream argument.
     */
    template<typename T, typename... Args>
    void emplace_reader(Args&&... args)
    {
        // This would probably work better with a lambda (better forwarding),
        // but GCC 4.8 has a bug with accessing parameter packs inside a
        // lambda.
        run_in_strand(detail::reference_bind(
                std::mem_fn(&stream::emplace_reader_callback<T, Args&&...>),
                this, std::forward<Args>(args)...));
    }

    /**
     * Return statistics about the stream. See the Python documentation.
     */
    stream_stats get_stats() const;

    /**
     * Stop the stream and block until all the readers have wound up. After
     * calling this there should be no more outstanding completion handlers
     * in the thread pool.
     *
     * In most cases subclasses should override @ref stop_received rather than
     * this function.
     */
    virtual void stop();
};

/**
 * Push packets found in a block of memory to a stream. Returns a pointer to
 * after the last packet found in the stream. Processing stops as soon as
 * after @ref decode_packet fails (because there is no way to find the next
 * packet after a corrupt one), but packets may still be rejected by the stream.
 *
 * The stream is @em not stopped.
 */
const std::uint8_t *mem_to_stream(stream_base &s, const std::uint8_t *ptr, std::size_t length);

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_STREAM_H
