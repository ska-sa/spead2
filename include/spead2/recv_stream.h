/* Copyright 2015, 2017-2019 SKA South Africa
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
#include <spead2/common_semaphore.h>

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
 *
 * @internal
 *
 * Avoiding deadlocks requires a careful design with several mutexes. It's
 * governed by the requirement that @ref heap_ready may block indefinitely, and
 * this must not block other functions. Thus, several mutexes are involved:
 *   - @ref queue_mutex: protects values only used by @ref add_packet. This
 *     may be locked for long periods.
 *   - @ref config_mutex: protects configuration. The protected values are
 *     copied into @ref add_packet_state prior to adding a batch of packets.
 *     It is mostly locked for reads.
 *   - @ref stats_mutex: protects stream statistics, and is mostly locked for
 *     writes (assuming the user is only occasionally checking the stats).
 *
 * While holding @ref config_mutex or @ref stats_mutex it is illegal to lock
 * any of the other mutexes.
 *
 * The public interface takes care of locking the appropriate mutexes. The
 * private member functions generally expect the caller to take locks.
 */
class stream_base
{
public:
    struct add_packet_state;

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
    const std::unique_ptr<storage_type[]> queue_storage;
    /// Number of entries in @ref buckets
    const std::size_t bucket_count;
    /// Right shift to map 64-bit unsigned to a bucket index
    const int bucket_shift;
    /// Pointer to the first heap in each bucket, or NULL
    const std::unique_ptr<queue_entry *[]> buckets;
    /// Position of the most recently added heap
    std::size_t head;

    /// Maximum number of live heaps permitted.
    const std::size_t max_heaps;
    /// Protocol bugs to be compatible with
    const bug_compat_mask bug_compat;

    /**
     * Mutex protecting the state of the queue. This includes
     * - @ref queue_storage
     * - @ref buckets
     * - @ref head
     * - @ref stopped
     */
    mutable std::mutex queue_mutex;

    /**
     * Mutex protecting configuration. This includes
     * - @ref allocator
     * - @ref memcpy
     * - @ref stop_on_stop_item
     * - @ref allow_unsized_heaps
     */
    mutable std::mutex config_mutex;

    /// Function used to copy heap payloads
    packet_memcpy_function memcpy;
    /// Whether to stop when a stream control stop item is received
    bool stop_on_stop_item = true;
    /// Whether to permit packets that don't have HEAP_LENGTH item
    bool allow_unsized_heaps = true;

    /// Memory allocator used by heaps.
    std::shared_ptr<memory_allocator> allocator;

    /// @ref stop_received has been called, either externally or by stream control
    bool stopped = false;

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
     * The heap might or might not be complete. The @ref queue_mutex will be
     * locked during this call, which will block @ref stop and @ref flush.
     */
    virtual void heap_ready(live_heap &&) {}

    /// Implementation of @ref flush that assumes the caller has locked @ref queue_mutex
    void flush_unlocked();

    /// Implementation of @ref stop that assumes the caller has locked @ref queue_mutex
    void stop_unlocked();

    /// Implementation of @ref add_packet_state::add_packet
    bool add_packet(add_packet_state &state, const packet_header &packet);

protected:
    mutable std::mutex stats_mutex;
    stream_stats stats;

    /**
     * Shut down the stream. This calls @ref flush_unlocked. Subclasses may
     * override this to achieve additional effects, but must chain to the base
     * implementation. It is guaranteed that it will only be called once.
     *
     * It is undefined what happens if @ref add_packet is called after a stream
     * is stopped.
     *
     * This is called with @ref queue_mutex locked. Users must not call this
     * function themselves; instead, call @ref stop.
     */
    virtual void stop_received();

public:
    /**
     * State for a batch of calls to @ref add_packet. Constructing this object
     * locks the stream's @ref queue_mutex.
     */
    struct add_packet_state
    {
        stream_base &owner;
        std::lock_guard<std::mutex> lock;    ///< Holds a lock on the owner's @ref queue_mutex

        // Copied from the stream, but unencumbered by locks/atomics
        packet_memcpy_function memcpy;
        std::shared_ptr<memory_allocator> allocator;
        bool stop_on_stop_item;
        bool allow_unsized_heaps;
        // Updates to the statistics
        std::uint64_t packets = 0;
        std::uint64_t complete_heaps = 0;
        std::uint64_t incomplete_heaps_evicted = 0;
        std::uint64_t single_packet_heaps = 0;
        std::uint64_t search_dist = 0;

        explicit add_packet_state(stream_base &owner);
        ~add_packet_state();

        bool is_stopped() const { return owner.stopped; }
        /// Indicate that the stream has stopped (e.g. because the remote peer disconnected)
        void stop() { owner.stop_unlocked(); }
        /**
         * Add a packet that was received, and which has been examined by @ref
         * decode_packet, and returns @c true if it is consumed. Even though @ref
         * decode_packet does some basic sanity-checking, it may still be rejected
         * by @ref live_heap::add_packet e.g., because it is a duplicate.
         *
         * It is an error to call this after the stream has been stopped.
         */
        bool add_packet(const packet_header &packet) { return owner.add_packet(*this, packet); }
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
    void set_memcpy(packet_memcpy_function memcpy);

    /// Set an alternative memcpy function for copying heap payload
    void set_memcpy(memcpy_function memcpy);

    /// Set builtin memcpy function to use for copying payload
    void set_memcpy(memcpy_function_id id);

    /// Set whether to stop the stream when a stop item is received
    void set_stop_on_stop_item(bool stop);

    /// Get whether to stop the stream when a stop item is received
    bool get_stop_on_stop_item() const;

    /// Set whether to allow heaps without HEAP_LENGTH
    void set_allow_unsized_heaps(bool allow);

    /// Get whether to allow heaps without HEAP_LENGTH
    bool get_allow_unsized_heaps() const;

    bug_compat_mask get_bug_compat() const { return bug_compat; }

    /// Flush the collection of live heaps, passing them to @ref heap_ready.
    void flush();

    /**
     * Stop the stream. This calls @ref stop_received.
     */
    void stop();

    /**
     * Return statistics about the stream. See the Python documentation.
     */
    stream_stats get_stats() const;
};

/**
 * Stream that is fed by subclasses of @ref reader.
 *
 * The public interface to this class is thread-safe.
 */
class stream : protected stream_base
{
private:
    friend class reader;

    /// Holder that just ensures that the thread pool doesn't vanish
    std::shared_ptr<thread_pool> thread_pool_holder;

    /// I/O service used by the readers
    boost::asio::io_service &io_service;

    /// Protects mutable state (@ref readers, @ref stop_readers, @ref lossy).
    mutable std::mutex reader_mutex;
    /**
     * Readers providing the stream data.
     */
    std::vector<std::unique_ptr<reader> > readers;

    /// Set to true to indicate that no new readers should be added
    bool stop_readers = false;

    /// True if any lossy reader has been added
    bool lossy = false;

    /// Ensure that @ref stop is only run once
    std::once_flag stop_once;

    /// Incremented by readers when they die
    semaphore readers_stopped;

    /* Prevent moving (copying is already impossible). Moving is not safe
     * because readers refer back to *this (it could potentially be added if
     * there is a good reason for it, but it would require adding a new
     * function to the reader interface.
     */
    stream(stream_base &&) = delete;
    stream &operator=(stream_base &&) = delete;

protected:
    virtual void stop_received() override;

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
    using stream_base::set_allow_unsized_heaps;
    using stream_base::get_allow_unsized_heaps;
    using stream_base::get_stats;

    explicit stream(io_service_ref io_service, bug_compat_mask bug_compat = 0, std::size_t max_heaps = default_max_heaps);
    virtual ~stream() override;

    boost::asio::io_service &get_io_service() { return io_service; }

    /**
     * Add a new reader by passing its constructor arguments, excluding
     * the initial @a stream argument.
     */
    template<typename T, typename... Args>
    void emplace_reader(Args&&... args)
    {
        std::lock_guard<std::mutex> lock(reader_mutex);
        // See comments in stop_impl for why we do this check
        if (!stop_readers)
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

    /**
     * Stop the stream and block until all the readers have wound up. After
     * calling this there should be no more outstanding completion handlers
     * in the thread pool.
     *
     * In most cases subclasses should override @ref stop_received rather than
     * this function. However, if @ref heap_ready can block indefinitely, this
     * function should be overridden to unblock it before calling the base
     * implementation.
     */
    virtual void stop();

    bool is_lossy() const;
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
