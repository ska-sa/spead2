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
#include <mutex>
#include <memory>
#include <spead2/recv_stream.h>
#include <spead2/recv_chunk_stream.h>

namespace spead2
{
namespace recv
{

/// Configuration for chunk_stream_group
class chunk_stream_group_config
{
public:
    /// Default value for @ref set_max_chunks
    static constexpr std::size_t default_max_chunks = chunk_stream_config::default_max_chunks;

private:
    std::size_t max_chunks = default_max_chunks;
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
    void ready_chunk(chunk_stream_state<chunk_manager_group> &state, chunk *c);
};

} // namespace detail

class chunk_stream_group_member;

/**
 * A holder for a collection of streams that share chunks.
 *
 * @todo write more documentation here
 */
class chunk_stream_group
{
private:
    friend class detail::chunk_manager_group;
    friend class chunk_stream_group_member;

    const chunk_stream_group_config config;

    std::mutex mutex; // Protects all the mutable state

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
     * References to the component streams that have not yet been stopped.
     *
     * Note that these are insufficient to actually keep the streams alive.
     * The stream_stop_received callback ensures that we don't end up with
     * dangling pointers.
     */
    std::set<chunk_stream_group_member *> streams;

    /**
     * Obtain the chunk with a given ID.
     *
     * This will shift the window if the chunk_id is beyond the tail. If the
     * chunk is too old, it will return @c nullptr. The reference count of the
     * returned chunk will be incremented.
     *
     * This function is thread-safe.
     */
    chunk *get_chunk(std::int64_t chunk_id, std::uintptr_t stream_id, std::uint64_t *batch_stats);

    /**
     * Decrement chunk reference count.
     *
     * If the reference count reaches zero, the chunk is passed to the ready
     * callback.
     *
     * This function is thread-safe.
     */
    void release_chunk(chunk *c, std::uint64_t *batch_stats);

    /// Version of release_chunk that does not take the lock
    void release_chunk_unlocked(chunk *c, std::uint64_t *batch_stats);

protected:
    /// Called by newly-constructed streams
    virtual void stream_added(chunk_stream_group_member &s);
    /**
     * Called when a stream stops (whether from the network or the user).
     *
     * The stream's @c queue_mutex is locked when this is called.
     */
    virtual void stream_stop_received(chunk_stream_group_member &s);
    /**
     * Called when the user stops (or destroys) a stream.
     *
     * This is called before the caller actually stops the stream, and without
     * the stream's @c queue_mutex.
     */
    virtual void stream_pre_stop(chunk_stream_group_member &s) {}

public:
    chunk_stream_group(const chunk_stream_group_config &config);
    virtual ~chunk_stream_group();

    /**
     * Stop all streams and release all chunks. This function must not be
     * called concurrently with creating or destroying streams, and no
     * new streams should be created after calling this.
     */
    virtual void stop();
};

/**
 * Single single within a group managed by @ref chunk_stream_group.
 */
class chunk_stream_group_member : private detail::chunk_stream_state<detail::chunk_manager_group>, public stream
{
    friend class detail::chunk_manager_group;

private:
    chunk_stream_group &group;  // TODO: redundant - also stored inside the manager

    virtual void heap_ready(live_heap &&) override;

public:
    using heap_metadata = detail::chunk_stream_state_base::heap_metadata;

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
     * Instances of this class must not outlive the group.
     *
     * @param io_service       I/O service (also used by the readers).
     * @param config           Basic stream configuration
     * @param chunk_config     Configuration for chunking
     * @param group            Group to which this stream belongs
     *
     * @throw invalid_argument if the place function pointer in @a chunk_config
     * has not been set.
     */
    chunk_stream_group_member(
        io_service_ref io_service,
        const stream_config &config,
        const chunk_stream_config &chunk_config,
        chunk_stream_group &group);

    using detail::chunk_stream_state_base::get_chunk_config;
    using detail::chunk_stream_state_base::get_heap_metadata;

    virtual void stop_received() override;
    virtual void stop() override;
    virtual ~chunk_stream_group_member() override;
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
 * When @ref stream::stop is called on any member stream, the ringbuffers
 * are both stopped, and readied chunks are diverted into a graveyard.
 * When @ref stop is called, the graveyard is emptied from the stream calling
 * @ref stop. This makes it safe to use chunks that can only safely be freed
 * from the caller's thread (e.g. a Python thread holding the GIL).
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
        DataRingbuffer &data_ring,
        FreeRingbuffer &free_ring,
        std::unique_ptr<chunk> &graveyard);

protected:
    virtual void stream_added(chunk_stream_group_member &s) override;
    virtual void stream_stop_received(chunk_stream_group_member &s) override;
    virtual void stream_pre_stop(chunk_stream_group_member &s) override;

public:
    chunk_stream_ring_group(
        const chunk_stream_group_config &group_config,
        std::shared_ptr<DataRingbuffer> data_ring,
        std::shared_ptr<FreeRingbuffer> free_ring);
    virtual void stop() override;

    ~chunk_stream_ring_group();
};

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::chunk_stream_ring_group(
    const chunk_stream_group_config &group_config,
    std::shared_ptr<DataRingbuffer> data_ring,
    std::shared_ptr<FreeRingbuffer> free_ring)
    : detail::chunk_ring_pair<DataRingbuffer, FreeRingbuffer>(std::move(data_ring), std::move(free_ring)),
    chunk_stream_group(adjust_group_config(this->data_ring, this->free_ring, this->graveyard))
{
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::stream_added(
    chunk_stream_group_member &s)
{
    chunk_stream_group::stream_added(s);
    this->data_ring.add_producer();
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::stream_stop_received(
    chunk_stream_group_member &s)
{
    chunk_stream_group::stream_stop_received(s);
    this->data_ring.remove_producer();
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::stream_pre_stop(
    chunk_stream_group_member &s)
{
    // Shut down the rings so that if the caller is no longer servicing them, it will
    // not lead to a deadlock during shutdown.
    this->data_ring.stop();
    this->free_ring.stop();
    chunk_stream_group::stream_pre_stop(s);
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::stop()
{
    // Stopping the first stream should do this anyway, but this ensures
    // they're stopped even if there are no streams
    this->data_ring.stop();
    this->free_ring.stop();
    chunk_stream_group::stop();
    this->graveyard.reset();  // Release chunks from the graveyard
}

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_stream_ring_group<DataRingbuffer, FreeRingbuffer>::~chunk_stream_ring_group()
{
    stop();
}

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_CHUNK_STREAM_GROUP
