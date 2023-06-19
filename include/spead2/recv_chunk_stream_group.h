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
#include <vector>
#include <mutex>
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

/**
 * A holder for a collection of streams that share chunks.
 *
 * @todo write more documentation here
 */
class chunk_stream_group
{
private:
    friend class detail::chunk_manager_group;

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

public:
    chunk_stream_group(const chunk_stream_group_config &config);
    ~chunk_stream_group();

    /**
     * Release all chunks. This function is thread-safe.
     */
    void flush_chunks();
};

/**
 * Single single within a group managed by @ref chunk_stream_group.
 */
class chunk_stream_group_member : private detail::chunk_stream_state<detail::chunk_manager_group>, public stream
{
    friend class detail::chunk_manager_group;

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

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_CHUNK_STREAM_GROUP
