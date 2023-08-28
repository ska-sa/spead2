/* Copyright 2023 National Research Foundation (SARAO)
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

#ifndef SPEAD2_RECV_CHUNK_STREAM_GROUP
#define SPEAD2_RECV_CHUNK_STREAM_GROUP

#include <cstddef>
#include <cstdint>
#include <set>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <boost/iterator/transform_iterator.hpp>
#include <spead2/recv_stream.h>
#include <spead2/recv_chunk_stream.h>

namespace spead2::recv
{

/// Configuration for chunk_stream_group
class chunk_stream_group_config
{
public:
    /// Default value for @ref set_max_chunks
    static constexpr std::size_t default_max_chunks = chunk_stream_config::default_max_chunks;

    /**
     * Eviction mode when it is necessary to advance the group window. See the
     * @verbatim embed:rst:inline :doc:`overview <recv-chunk-group>` @endverbatim
     * for more details.
     */
    enum class eviction_mode
    {
        LOSSY,    ///< force streams to release incomplete chunks
        LOSSLESS  ///< a chunk will only be marked ready when all streams have marked it ready
    };

private:
    std::size_t max_chunks = default_max_chunks;
    eviction_mode eviction_mode_ = eviction_mode::LOSSY;
    chunk_allocate_function allocate;
    chunk_ready_function ready;

public:
    /**
     * Set the maximum number of chunks that can be live at the same time.
     * A value of 1 means that heaps must be received in order: once a
     * chunk is started, no heaps from a previous chunk will be accepted.
     *
     * @throw std::invalid_argument if @a max_chunks is 0.
     */
    chunk_stream_group_config &set_max_chunks(std::size_t max_chunks);
    /// Return the maximum number of chunks that can be live at the same time.
    std::size_t get_max_chunks() const { return max_chunks; }

    /// Set chunk eviction mode. See @ref eviction_mode.
    chunk_stream_group_config &set_eviction_mode(eviction_mode eviction_mode_);
    /// Return the current eviction mode
    eviction_mode get_eviction_mode() const { return eviction_mode_; }

    /// Set the function used to allocate a chunk.
    chunk_stream_group_config &set_allocate(chunk_allocate_function allocate);
    /// Get the function used to allocate a chunk.
    const chunk_allocate_function &get_allocate() const { return allocate; }

    /// Set the function that is provided with completed chunks.
    chunk_stream_group_config &set_ready(chunk_ready_function ready);
    /// Get the function that is provided with completed chunks.
    const chunk_ready_function &get_ready() const { return ready; }
};

class chunk_stream_group;

namespace detail
{

class chunk_manager_group
{
private:
    chunk_stream_group &group;

public:
    explicit chunk_manager_group(chunk_stream_group &group);

    std::uint64_t *get_batch_stats(chunk_stream_state<chunk_manager_group> &state) const;
    chunk *allocate_chunk(chunk_stream_state<chunk_manager_group> &state, std::int64_t chunk_id);
    void ready_chunk(chunk_stream_state<chunk_manager_group> &, chunk *) {}
    void head_updated(chunk_stream_state<chunk_manager_group> &state, std::uint64_t head_chunk);
};

} // namespace detail

class chunk_stream_group_member;

/**
 * A holder for a collection of streams that share chunks. The group owns the
 * component streams, and takes care of stopping and destroying them when the
 * group is stopped or destroyed.
 *
 * It presents an interface similar to @c std::vector for observing the set
 * of attached streams.
 *
 * The public interface must only be called from one thread at a time, and
 * all streams must be added before any readers are attached to them.
 */
class chunk_stream_group
{
private:
    friend class detail::chunk_manager_group;
    friend class chunk_stream_group_member;

    const chunk_stream_group_config config;

    std::mutex mutex; ///< Protects all the mutable state
    /// Notified when the reference count of a chunk reaches zero
    std::condition_variable ready_condition;

    /**
     * Circular buffer of chunks under construction.
     *
     * Ownership of the chunks is shared between the group and the member
     * streams, but reference counting is manual (rather than using
     * std::shared_ptr) so that the reference count can be embedded in the
     * object, and to facilitate code sharing with @ref chunk_stream.
     */
    detail::chunk_window chunks;

    /**
     * The component streams.
     *
     * This is protected by the mutex, except that read-only access is always
     * permitted in methods called by the user. This is safe because writes
     * only happen in methods called by the user (@ref emplace_back), and the
     * user is required to serialise their calls.
     */
    std::vector<std::unique_ptr<chunk_stream_group_member>> streams;

    /**
     * Copy of the head chunk ID from each stream. This copy is protected by
     * the group's mutex rather than the streams'.
     *
     * The minimum element must always be equal to @c chunks.get_head_chunk().
     */
    std::vector<std::uint64_t> head_chunks;

    /**
     * Last value passed to all streams' async_flush_until.
     */
    std::uint64_t last_flush_until = 0;

    /**
     * Obtain the chunk with a given ID.
     *
     * This will shift the window if the chunk_id is beyond the tail. If the
     * chunk is too old, it will return @c nullptr. The reference count of the
     * returned chunk will be incremented.
     *
     * This function is thread-safe.
     */
    chunk *get_chunk(std::uint64_t chunk_id, std::uintptr_t stream_id, std::uint64_t *batch_stats);

    /**
     * Called by a stream to report movement in its head pointer. This function
     * takes the group mutex.
     */
    void stream_head_updated(chunk_stream_group_member &s, std::uint64_t head_chunk);

    /**
     * Pass a chunk to the user-provided ready function. The caller is
     * responsible for ensuring that the chunk is no longer in use.
     *
     * The caller must hold the group mutex.
     */
    void ready_chunk(chunk *c, std::uint64_t *batch_stats);

    // Helper classes for implementing iterators
    template<typename T>
    class dereference
    {
    public:
        decltype(auto) operator()(const T &ptr) const { return *ptr; }
    };

    template<typename T>
    class dereference_const
    {
    public:
        decltype(auto) operator()(const T &ptr) const { return *ptr; }
    };

protected:
    /**
     * Called by @ref emplace_back for newly-constructed streams. The group's
     * mutex is held when this is called.
     */
    virtual void stream_added(chunk_stream_group_member &) {}
    /**
     * Called when a stream stops (whether from the network or the user).
     *
     * The stream's @c queue_mutex is locked when this is called.
     */
    virtual void stream_stop_received(chunk_stream_group_member &) {}

public:
    using iterator = boost::transform_iterator<
        dereference<std::unique_ptr<chunk_stream_group_member>>,
        std::vector<std::unique_ptr<chunk_stream_group_member>>::iterator
    >;
    using const_iterator = boost::transform_iterator<
        dereference_const<std::unique_ptr<chunk_stream_group_member>>,
        std::vector<std::unique_ptr<chunk_stream_group_member>>::const_iterator
    >;

    explicit chunk_stream_group(const chunk_stream_group_config &config);
    virtual ~chunk_stream_group();

    const chunk_stream_group_config &get_config() const { return config; }

    /// Add a new stream
    chunk_stream_group_member &emplace_back(
        io_service_ref io_service,
        const stream_config &config,
        const chunk_stream_config &chunk_config);

    /// Add a new stream, possibly of a subclass
    template<typename T, typename... Args>
    T &emplace_back(Args&&... args);

    /**
     * @name Vector-like access to the streams.
     * Iterator invalidation rules are the same as for @c std::vector i.e.,
     * modifying the set of streams invalidates iterators.
     * @{
     */
    /// Number of streams
    std::size_t size() const { return streams.size(); }
    /// Whether there are any streams
    bool empty() const { return streams.empty(); }
    /// Get the stream at a given index
    chunk_stream_group_member &operator[](std::size_t index) { return *streams[index]; }
    /// Get the stream at a given index
    const chunk_stream_group_member &operator[](std::size_t index) const { return *streams[index]; }
    /// Get an iterator to the first stream
    iterator begin() noexcept;
    /// Get an iterator past the last stream
    iterator end() noexcept;
    /// Get an iterator to the first stream
    const_iterator begin() const noexcept;
    /// Get a const iterator past the last stream
    const_iterator end() const noexcept;
    /// Get an iterator to the first stream
    const_iterator cbegin() const noexcept;
    /// Get a const iterator past the last stream
    const_iterator cend() const noexcept;
    /**
     * @}
     */

    /// Stop all streams and release all chunks.
    virtual void stop();
};

/**
 * Single single within a group managed by @ref chunk_stream_group.
 */
class chunk_stream_group_member : private detail::chunk_stream_state<detail::chunk_manager_group>, public stream
{
    friend class detail::chunk_manager_group;
    friend class chunk_stream_group;

private:
    chunk_stream_group &group;  // TODO: redundant - also stored inside the manager
    const std::size_t group_index;  ///< Position of the chunk within the group

    virtual void heap_ready(live_heap &&) override;

    /**
     * Flush all chunks with an ID strictly less than @a chunk_id.
     *
     * This function returns immediately, and the work is done later on the
     * io_service. It is safe to call from any thread.
     */
    void async_flush_until(std::uint64_t chunk_id);

protected:
    /**
     * Constructor.
     *
     * This class passes a modified @a config to the base class constructor.
     * See @ref chunk_stream for more information.
     *
     * The @link chunk_stream_config::set_allocate allocate@endlink and
     * @link chunk_stream_config::set_ready ready@endlink callbacks are
     * ignored, and the group's callbacks are used instead.
     *
     * @param group            Group to which this stream belongs
     * @param group_index      Position of this stream within the group
     * @param io_service       I/O service (also used by the readers).
     * @param config           Basic stream configuration
     * @param chunk_config     Configuration for chunking
     *
     * @throw invalid_argument if the place function pointer in @a chunk_config
     * has not been set.
     */
    chunk_stream_group_member(
        chunk_stream_group &group,
        std::size_t group_index,
        io_service_ref io_service,
        const stream_config &config,
        const chunk_stream_config &chunk_config);

    /**
     * Stop just this stream. This does the real work of stopping the stream,
     * whereas the public @ref stop function stops the entire group.
     *
     * This should only be called from @ref chunk_stream_group::stop.
     */
    virtual void stop1();

public:
    using heap_metadata = detail::chunk_stream_state_base::heap_metadata;

    using detail::chunk_stream_state_base::get_chunk_config;
    using detail::chunk_stream_state_base::get_heap_metadata;

    virtual void stop_received() override;
    virtual void stop() override;
    /* Note: most stream classes have a destructor that calls stop(),
     * but that's not required nor safe for this class: stop() calls
     * group.stop(), but the stream is only destroyed as part of destroying
     * the group. Instead, the group's destructor ensures that stop1 is
     * called.
     */
};

/**
 * Wrapper around @ref chunk_stream_group that uses ringbuffers to manage
 * chunks.
 *
 * When a fresh chunk is needed, it is retrieved from a ringbuffer of free
 * chunks (the "free ring"). When a chunk is flushed, it is pushed to a
 * "data ring". These may be shared between groups, but both will be
 * stopped as soon as any of the members streams are stopped. The intended use
 * case is parallel groups that are started and stopped together.
 *
 * When the group is stopped, the ringbuffers are both stopped, and readied
 * chunks are diverted into a graveyard. The graveyard is then emptied from
 * the thread calling @ref stop. This makes it safe to use chunks that can only
 * safely be freed from the caller's thread (e.g. a Python thread holding the
 * GIL).
 */
template<typename DataRingbuffer = ringbuffer<std::unique_ptr<chunk>>,
         typename FreeRingbuffer = ringbuffer<std::unique_ptr<chunk>>>
class chunk_stream_ring_group
: public detail::chunk_ring_pair<DataRingbuffer, FreeRingbuffer>, public chunk_stream_group
{
private:
    /// Create a new @ref chunk_stream_group_config that uses the ringbuffers
    static chunk_stream_group_config adjust_group_config(
        const chunk_stream_group_config &config,
        detail::chunk_ring_pair<DataRingbuffer, FreeRingbuffer> &ring_pair);

protected:
    virtual void stream_added(chunk_stream_group_member &s) override;
    virtual void stream_stop_received(chunk_stream_group_member &s) override;

public:
    chunk_stream_ring_group(
        const chunk_stream_group_config &group_config,
        std::shared_ptr<DataRingbuffer> data_ring,
        std::shared_ptr<FreeRingbuffer> free_ring);
    virtual void stop() override;

    ~chunk_stream_ring_group();
};


template<typename T, typename... Args>
T &chunk_stream_group::emplace_back(Args&&... args)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (chunks.get_tail_chunk() != 0 || last_flush_until != 0)
    {
        throw std::runtime_error("Cannot add a stream after group has started receiving data");
    }
    std::unique_ptr<T> stream(new T(
        *this, streams.size(), std::forward<Args>(args)...));
    T &ret = *stream;
    streams.push_back(std::move(stream));
    head_chunks.push_back(0);
    stream_added(ret);
    return ret;
}

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::chunk_stream_ring_group(
    const chunk_stream_group_config &group_config,
    std::shared_ptr<DataRingbuffer> data_ring,
    std::shared_ptr<FreeRingbuffer> free_ring)
    : detail::chunk_ring_pair<DataRingbuffer, FreeRingbuffer>(std::move(data_ring), std::move(free_ring)),
    chunk_stream_group(adjust_group_config(group_config, *this))
{
}

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_stream_group_config chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::adjust_group_config(
    const chunk_stream_group_config &config,
    detail::chunk_ring_pair<DataRingbuffer, FreeRingbuffer> &ring_pair)
{
    chunk_stream_group_config new_config = config;
    new_config.set_allocate(ring_pair.make_allocate());
    new_config.set_ready(ring_pair.make_ready(config.get_ready()));
    return new_config;
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::stream_added(
    chunk_stream_group_member &s)
{
    chunk_stream_group::stream_added(s);
    this->data_ring->add_producer();
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::stream_stop_received(
    chunk_stream_group_member &s)
{
    chunk_stream_group::stream_stop_received(s);
    this->data_ring->remove_producer();
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::stop()
{
    // Shut down the rings so that if the caller is no longer servicing them, it will
    // not lead to a deadlock during shutdown.
    this->data_ring->stop();
    this->free_ring->stop();
    chunk_stream_group::stop();
    this->graveyard.reset();  // Release chunks from the graveyard
}

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::~chunk_stream_ring_group()
{
    stop();
}

} // namespace spead2::recv

#endif // SPEAD2_RECV_CHUNK_STREAM_GROUP
