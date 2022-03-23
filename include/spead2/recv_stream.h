/* Copyright 2015, 2017-2021 National Research Foundation (SARAO)
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
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <utility>
#include <functional>
#include <future>
#include <mutex>
#include <atomic>
#include <iterator>
#include <type_traits>
#include <boost/asio.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <libdivide.h>
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

/// Registration information about a statistic counter.
class stream_stat_config
{
public:
    /**
     * Type for a statistic.
     *
     * All statistics are integral, but the mode determines how values are merged.
     */
    enum class mode
    {
        COUNTER,    ///< Merge values by addition
        MAXIMUM     ///< Merge values by taking the larger one
    };

private:
    const std::string name;
    const mode mode_;

public:
    explicit stream_stat_config(std::string name, mode mode_ = mode::COUNTER);

    /// Get the name passed to the constructor
    const std::string &get_name() const { return name; }
    /// Get the mode passed to the constructor
    mode get_mode() const { return mode_; }
    /// Combine two samples according to the mode.
    std::uint64_t combine(std::uint64_t a, std::uint64_t b) const;
};

/* Comparison operators for stream_stat_config is used to check whether two
 * instances of stream_stat have the same config and hence can be sensibly
 * combined.
 */
bool operator==(const stream_stat_config &a, const stream_stat_config &b);
bool operator!=(const stream_stat_config &a, const stream_stat_config &b);

/// Constants for indexing @ref stream_stats by index
namespace stream_stat_indices
{

static constexpr std::size_t heaps = 0;
static constexpr std::size_t incomplete_heaps_evicted = 1;
static constexpr std::size_t incomplete_heaps_flushed = 2;
static constexpr std::size_t packets = 3;
static constexpr std::size_t batches = 4;
static constexpr std::size_t max_batch = 5;
static constexpr std::size_t single_packet_heaps = 6;
static constexpr std::size_t search_dist = 7;
static constexpr std::size_t worker_blocked = 8;
static constexpr std::size_t custom = 9;  ///< Index for first user-defined statistic

} // namespace stream_stat_indices

namespace detail
{

/* Implementation details of stream_stats::iterator and
 * stream_stats::const_iterator. It zips together the names and values of the
 * statistics. Because dereferencing returns temporaries rather than lvalue
 * references, this is actually an input iterator (in pre-C++20 terminology),
 * although it does support random traversal.
 *
 * T is either stream_stats or const stream_stats
 * V is the value type of the iterator (a pair of name and value)
 *
 * boost::iterator_facade simplifies the implementation by filling in all the
 * different member types and functions expected of a conforming iterator.
 */
template<typename T, typename V>  // T is either stream_stats or const stream_stats; V is a pair
class stream_stats_iterator : public boost::iterator_facade<
    stream_stats_iterator<T, V>,
    V, // value type
    boost::random_access_traversal_tag,
    V> // reference type
{
private:
    friend class boost::iterator_core_access;
    template<typename T2, typename V2> friend class stream_stats_iterator;

    T *owner = nullptr;
    std::size_t index = 0;

    V dereference() const
    {
        return V(owner->get_config()[index].get_name(), (*owner)[index]);
    }

    template<typename T2, typename V2>
    bool equal(const stream_stats_iterator<T2, V2> &other) const
    {
        return owner == other.owner && index == other.index;
    }

    void increment() { index++; }
    void decrement() { index--; }
    void advance(std::ptrdiff_t n) { index += n; }

    template<typename T2, typename V2>
    std::ptrdiff_t distance_to(const stream_stats_iterator<T2, V2> &other) const
    {
        return std::ptrdiff_t(other.index) - std::ptrdiff_t(index);
    }

public:
    stream_stats_iterator() = default;
    explicit stream_stats_iterator(T &owner, std::size_t index = 0) : owner(&owner), index(index) {}

    // This is a template constructor to allow iterator to be converted to const_iterator
    template<typename T2, typename V2,
             typename = typename std::enable_if<std::is_convertible<T2 *, T *>::value>::type>
    stream_stats_iterator(const stream_stats_iterator<T2, V2> &other)
        : owner(other.owner), index(other.index) {}
};

} // namespace detail

/**
 * Statistics about a stream. Both a vector-like interface (indexing and @ref
 * size) and a map-like interface (with iterators and @ref find) are provided.
 * The iterators support random traversal, but are not technically random
 * access iterators because dereferencing does not return a reference.
 *
 * The public members provide direct access to the core statistics, for
 * backwards compatibility. New code is advised to use the other interfaces.
 * Indices for core statistics are available in @ref stream_stat_indices.
 */
class stream_stats
{
private:
    std::shared_ptr<std::vector<stream_stat_config>> config;
    std::vector<std::uint64_t> values;

public:
    // These are all to make the container look like a std::unordered_map
    using key_type = std::string;
    using mapped_type = std::uint64_t;
    using value_type = std::pair<const std::string, std::uint64_t>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using iterator = detail::stream_stats_iterator<stream_stats, std::pair<const std::string &, std::uint64_t &>>;
    using const_iterator = detail::stream_stats_iterator<const stream_stats, const std::pair<const std::string &, const std::uint64_t &>>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /// Construct with the default set of statistics, and all zero values
    stream_stats();
    /// Construct with all zero values
    explicit stream_stats(std::shared_ptr<std::vector<stream_stat_config>> config);
    /// Construct with provided values
    stream_stats(std::shared_ptr<std::vector<stream_stat_config>> config,
                 std::vector<std::uint64_t> values);

    /* Copy constructor and copy assignment need to be implemented manually
     * because of the embedded references. This will suppress the implicit
     * move assignment (which would also be unsafe). The implicit move
     * constructor is safe so we default it.
     */
    stream_stats(const stream_stats &other);
    stream_stats &operator=(const stream_stats &other);
    stream_stats(stream_stats &&other) = default;

    /// Get the configuration of the statistics
    const std::vector<stream_stat_config> &get_config() const { return *config; }

    /// Whether the container is empty
    bool empty() const { return size() == 0; }
    /// Get the number of statistics
    std::size_t size() const { return values.size(); }
    /**
     * Access a statistic by index. If index is out of range, behaviour is undefined.
     */
    std::uint64_t &operator[](std::size_t index) { return values[index]; }
    /**
     * Access a statistic by index. If index is out of range, behaviour is undefined.
     */
    const std::uint64_t &operator[](std::size_t index) const { return values[index]; }
    /**
     * Access a statistic by index.
     *
     * @throw std::out_of_range if index is out of range.
     */
    std::uint64_t &at(std::size_t index) { return values.at(index); }
    /**
     * Access a statistic by index.
     *
     * @throw std::out_of_range if index is out of range.
     */
    const std::uint64_t &at(std::size_t index) const { return values.at(index); }

    /**
     * Access a statistic by name.
     *
     * @throw std::out_of_range if @a name is not the name of a statistic
     */
    std::uint64_t &operator[](const std::string &name);
    /**
     * Access a statistic by name.
     *
     * @throw std::out_of_range if @a name is not the name of a statistic
     */
    const std::uint64_t &operator[](const std::string &name) const;
    /**
     * Access a statistic by name.
     *
     * @throw std::out_of_range if @a name is not the name of a statistic
     */
    std::uint64_t &at(const std::string &name);
    /**
     * Access a statistic by name.
     *
     * @throw std::out_of_range if @a name is not the name of a statistic
     */
    const std::uint64_t &at(const std::string &name) const;

    const_iterator cbegin() const noexcept { return const_iterator(*this); }
    const_iterator cend() const noexcept { return const_iterator(*this, size()); }
    iterator begin() noexcept { return iterator(*this); }
    iterator end() noexcept { return iterator(*this, size()); }
    const_iterator begin() const noexcept { return cbegin(); }
    const_iterator end() const noexcept { return cend(); }

    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }
    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator rbegin() const noexcept { return crbegin(); }
    const_reverse_iterator rend() const noexcept { return crend(); }
    /**
     * Find element with the given name. If not found, returns @c end().
     */
    iterator find(const std::string &name);
    /**
     * Find element with the given name. If not found, returns @c end().
     */
    const_iterator find(const std::string &name) const;
    /**
     * Return the number of elements matching @a name (0 or 1).
     */
    std::size_t count(const std::string &name) const;

    // References to core statistics in values (for backwards compatibility only).
    std::uint64_t &heaps;
    std::uint64_t &incomplete_heaps_evicted;
    std::uint64_t &incomplete_heaps_flushed;
    std::uint64_t &packets;
    std::uint64_t &batches;
    std::uint64_t &worker_blocked;
    std::uint64_t &max_batch;
    std::uint64_t &single_packet_heaps;
    std::uint64_t &search_dist;

    /**
     * Combine two sets of statistics. Each statistic is combined according to
     * its mode.
     *
     * @throw std::invalid_argument if @a other has a different list of statistics
     * @see stream_stat_config::mode
     */
    stream_stats operator+(const stream_stats &other) const;
    /**
     * Combine another set of statistics with this one. Each statistic is
     * combined according to its mode.
     *
     * @throw std::invalid_argument if @a other has a different list of statistics
     * @see stream_stat_config::mode
     */
    stream_stats &operator+=(const stream_stats &other);
};

/**
 * Parameters for a receive stream.
 */
class stream_config
{
    friend class stream_base;
public:
    static constexpr std::size_t default_max_heaps = 4;

private:
    /// Maximum number of live heaps permitted per substream
    std::size_t max_heaps = default_max_heaps;
    /// Number of substreams
    std::size_t substreams = 1;
    /// Protocol bugs to be compatible with
    bug_compat_mask bug_compat = 0;

    /// Function used to copy heap payloads
    packet_memcpy_function memcpy;
    /// Memory allocator used by heaps.
    std::shared_ptr<memory_allocator> allocator;
    /// Whether to stop when a stream control stop item is received
    bool stop_on_stop_item = true;
    /// Whether to permit packets that don't have HEAP_LENGTH item
    bool allow_unsized_heaps = true;
    /// Whether to accept packets out-of-order for a single heap
    bool allow_out_of_order = false;
    /// A user-defined identifier for a stream
    std::uintptr_t stream_id = 0;
    /// Statistics (includes the built-in ones)
    std::shared_ptr<std::vector<stream_stat_config>> stats;

public:
    stream_config();

    /**
     * Set maximum number of partial heaps that can be live at one time
     * (per substream). This affects how intermingled heaps can be (due to
     * out-of-order packet delivery) before heaps get dropped.
     */
    stream_config &set_max_heaps(std::size_t max_heaps);
    /// Get maximum number of partial heaps that can be live at one time.
    std::size_t get_max_heaps() const { return max_heaps; }

    /**
     * Set number of substreams. The substream is determined by taking the
     * heap cnt modulo the number of substreams. The value set by
     * @ref set_max_heaps applies independently for each substream.
     */
    stream_config &set_substreams(std::size_t substreams);
    /// Get number of substreams.
    std::size_t get_substreams() const { return substreams; }

    /// Set an allocator to use for allocating heap memory.
    stream_config &set_memory_allocator(std::shared_ptr<memory_allocator> allocator);
    /// Get allocator for allocating heap memory.
    const std::shared_ptr<memory_allocator> &get_memory_allocator() const
    {
        return allocator;
    }

    /// Set an alternative memcpy function for copying heap payload.
    stream_config &set_memcpy(packet_memcpy_function memcpy);

    /// Set an alternative memcpy function for copying heap payload.
    stream_config &set_memcpy(memcpy_function memcpy);

    /// Set builtin memcpy function to use for copying heap payload.
    stream_config &set_memcpy(memcpy_function_id id);

    /// Get memcpy function for copying heap payload.
    const packet_memcpy_function &get_memcpy() const { return memcpy; }

    /// Set whether to stop the stream when a stop item is received.
    stream_config & set_stop_on_stop_item(bool stop);

    /// Get whether to stop the stream when a stop item is received.
    bool get_stop_on_stop_item() const { return stop_on_stop_item; }

    /// Set whether to allow heaps without HEAP_LENGTH
    stream_config &set_allow_unsized_heaps(bool allow);

    /// Get whether to allow heaps without HEAP_LENGTH
    bool get_allow_unsized_heaps() const { return allow_unsized_heaps; }

    /// Set whether to allow out-of-order packets within a heap
    stream_config &set_allow_out_of_order(bool allow);

    /// Get whether to allow out-of-order packets within a heap
    bool get_allow_out_of_order() const { return allow_out_of_order; }

    /// Set bug compatibility flags.
    stream_config &set_bug_compat(bug_compat_mask bug_compat);

    /// Get bug compatibility flags.
    bug_compat_mask get_bug_compat() const { return bug_compat; }

    /// Set a stream ID
    stream_config &set_stream_id(std::uintptr_t stream_id);

    /// Get the stream ID
    std::uintptr_t get_stream_id() const { return stream_id; }

    /**
     * Add a new custom statistic. Returns the index to use with @ref stream_stats.
     *
     * @throw std::invalid_argument if @a name already exists.
     */
    std::size_t add_stat(
        std::string name,
        stream_stat_config::mode mode = stream_stat_config::mode::COUNTER);

    /// Get the stream statistics (including the core ones)
    const std::vector<stream_stat_config> &get_stats() const { return *stats; }

    /**
     * Helper to get the index of a specific statistic.
     *
     * @throw std::out_of_range if @a name is not a known statistic.
     */
    std::size_t get_stat_index(const std::string &name) const;

    /**
     * The index that will be returned by the next call to @ref add_stat.
     */
    std::size_t next_stat_index() const { return stats->size(); }
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
 *   incomplete; or
 * - @c allow_out_of_order is false and we have received a packet from the
 *   heap that is not the next expected one; or
 * - The stream is stopped.
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
 * When using multiple substreams, each substream has its own circular queue;
 * all the queues are held consecutively in a single storage allocation, but
 * each has a separate head pointer that wraps within its own portion of the
 * storage.
 *
 * Avoiding deadlocks requires a careful design with several mutexes. It's
 * governed by the requirement that @ref heap_ready may block indefinitely, and
 * this must not block other functions. Thus, several mutexes are involved:
 *   - @ref queue_mutex: protects values only used by @ref add_packet. This
 *     may be locked for long periods.
 *   - @ref stats_mutex: protects stream statistics, and is mostly locked for
 *     writes (assuming the user is only occasionally checking the stats).
 *
 * While holding @ref stats_mutex it is illegal to lock any other mutexes.
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

    // Per-substream data
    struct substream
    {
        /// Start of this substream within the queue
        std::size_t start;
        /// Position of the most recently-added heap
        std::size_t head;
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
    /**
     * Per-substream data. There is one extra entry to indicate the end
     * position of the last substream.
     */
    const std::unique_ptr<substream[]> substreams;
    /// Fast division by number of substreams
    libdivide::divider<item_pointer_t> substream_div;

    /// Stream configuration
    const stream_config config;

protected:
    /**
     * Mutex protecting the state of the queue. This includes
     * - @ref queue_storage
     * - @ref buckets
     * - @ref head
     * - @ref stopped
     *
     * Subclasses may use it to protect additional state. It is guaranteed to
     * be locked when @ref heap_ready is called.
     */
    mutable std::mutex queue_mutex;

private:
    /// @ref stop_received has been called, either externally or by stream control
    bool stopped = false;

    /// Compute bucket number for a heap cnt
    std::size_t get_bucket(item_pointer_t heap_cnt) const;

    /// Compute substream from a heap cnt
    std::size_t get_substream(item_pointer_t heap_cnt) const;

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
    std::vector<std::uint64_t> stats;

    /**
     * Statistics for the current batch. These are protected by queue_mutex
     * rather than stats_mutex. When the batch ends they are merged into
     * @ref stats. User code can safely update these stats from
     * within @ref stream::heap_ready, custom allocators and packet memcpy
     * functions. Only the custom statistics should be updated; it is
     * not guaranteed that built-in stats in this vector will be seen.
     */
    std::vector<std::uint64_t> batch_stats;

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

    /**
     * Constructor.
     *
     * @param config       Stream configuration
     */
    explicit stream_base(const stream_config &config = stream_config());
    virtual ~stream_base();

    /// Get the stream's configuration
    const stream_config &get_config() const { return config; }

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
    using stream_base::get_config;
    using stream_base::get_stats;

    explicit stream(io_service_ref io_service, const stream_config &config = stream_config());
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
