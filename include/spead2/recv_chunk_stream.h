/* Copyright 2021 National Research Foundation (SARAO)
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

#ifndef SPEAD2_RECV_CHUNK_STREAM
#define SPEAD2_RECV_CHUNK_STREAM

#include <memory>
#include <vector>
#include <functional>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <spead2/common_defines.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/recv_packet.h>
#include <spead2/recv_stream.h>

namespace spead2
{
namespace recv
{

/// Storage for a chunk with metadata
class chunk
{
public:
    /// Chunk ID
    std::int64_t chunk_id = -1;
    /// Stream ID of the stream from which the chunk originated
    std::uintptr_t stream_id = 0;
    /// Flag array indicating which heaps have been received (one byte per heap)
    memory_allocator::pointer present;
    /// Number of elements in @ref present
    std::size_t present_size = 0;
    /// Chunk payload
    memory_allocator::pointer data;

    chunk() = default;
    // These need to be explicitly declared, because there is an explicit destructor.
    chunk(chunk &&other) = default;
    chunk &operator=(chunk &&other) = default;

    // Allow chunks to be polymorphic
    virtual ~chunk() = default;
};

/**
 * Data passed to @ref chunk_place_function. This structure is designed to be
 * a plain C structure that can easily be handled by language bindings. As far
 * as possible, new fields will be added to the end but existing fields will
 * be retained, to preserve ABI compatibility.
 */
struct chunk_place_data
{
    const std::uint8_t *packet;      ///< Pointer to the original packet data
    std::size_t packet_size;         ///< Number of bytes referenced by @ref packet
    const s_item_pointer_t *items;   ///< Values of requested item pointers
    /// Chunk ID (output). Set to -1 (or leave unmodified) to discard the heap.
    std::int64_t chunk_id;
    std::size_t heap_index;          ///< Number of this heap within the chunk (output)
    std::size_t heap_offset;         ///< Byte offset of this heap within the chunk payload (output)
    std::uint64_t *batch_stats;      ///< Pointer to staging area for statistics
    // Note: when adding new fields, remember to update src/spead2/recv/numba.py
};

/**
 * Callback to determine where a heap is placed in the chunk stream.
 *
 * @param data       Pointer to the input and output arguments.
 * @param data_size  <code>sizeof(chunk_place_data)</code> at the time spead2 was compiled
 *
 * @see chunk_place_data
 */
typedef std::function<void(chunk_place_data *data, std::size_t data_size)> chunk_place_function;

/**
 * Callback to obtain storage for a new chunk. It does not need to populate
 * @ref chunk::chunk_id.
 */
// If this is updated, update doc/cpp-recv-chunk.rst as well. It's not
// documented with breathe due to a Doxygen bug.
typedef std::function<std::unique_ptr<chunk>(std::int64_t chunk_id, std::uint64_t *batch_stats)> chunk_allocate_function;

/**
 * Callback to receive a completed chunk. It takes ownership of the chunk.
 */
typedef std::function<void(std::unique_ptr<chunk> &&, std::uint64_t *batch_stats)> chunk_ready_function;

/**
 * Parameters for a @ref chunk_stream.
 */
class chunk_stream_config
{
public:
    /// Default value for @ref set_max_chunks
    static constexpr std::size_t default_max_chunks = 2;

private:
    std::vector<item_pointer_t> item_ids;
    std::size_t max_chunks = default_max_chunks;

    chunk_place_function place;
    chunk_allocate_function allocate;
    chunk_ready_function ready;

    std::size_t packet_presence_payload_size = 0;

public:
    /**
     * Specify the items whose immediate values should be passed to the
     * place function (see @ref chunk_place_function).
     */
    chunk_stream_config &set_items(const std::vector<item_pointer_t> &item_ids);
    /// Get the items set with @ref set_items.
    const std::vector<item_pointer_t> &get_items() const { return item_ids; }

    /**
     * Set the maximum number of chunks that can be live at the same time.
     * A value of 1 means that heaps must be received in order: once a
     * chunk is started, no heaps from a previous chunk will be accepted.
     *
     * @throw std::invalid_argument if @a max_chunks is 0.
     */
    chunk_stream_config &set_max_chunks(std::size_t max_chunks);
    /// Return the maximum number of chunks that can be live at the same time.
    std::size_t get_max_chunks() const { return max_chunks; }

    /// Set the function used to determine the chunk of each heap and its placement within the chunk.
    chunk_stream_config &set_place(chunk_place_function place);
    /// Get the function used to determine the chunk of each heap and its placement within the chunk.
    const chunk_place_function &get_place() const { return place; }

    /// Set the function used to allocate a chunk.
    chunk_stream_config &set_allocate(chunk_allocate_function allocate);
    /// Get the function used to allocate a chunk.
    const chunk_allocate_function &get_allocate() const { return allocate; }

    /// Set the function that is provided with completed chunks.
    chunk_stream_config &set_ready(chunk_ready_function ready);
    /// Get the function that is provided with completed chunks.
    const chunk_ready_function &get_ready() const { return ready; }

    /**
     * Enable the packet presence feature. The payload offset of each
     * packet is divided by @a payload_size and added to the heap index
     * before indexing @ref spead2::recv::chunk::present.
     *
     * @throw std::invalid_argument if @a payload_size is zero.
     */
    chunk_stream_config &enable_packet_presence(std::size_t payload_size);

    /**
     * Disable the packet presence feature enabled by
     * @ref enable_packet_presence.
     */
    chunk_stream_config &disable_packet_presence();

    /**
     * Retrieve the @c payload_size if packet presence is enabled, or 0 if not.
     */
    std::size_t get_packet_presence_payload_size() const { return packet_presence_payload_size; }
};

namespace detail
{

class chunk_stream_allocator;

/**
 * Base class that holds the internal state of @ref
 * spead2::recv::chunk_stream.
 *
 * This is split into a separate class to avoid some initialisation ordering
 * problems: it is constructed before the @ref spead2::recv::stream base class,
 * allowing the latter to use function objects that reference this class.
 */
class chunk_stream_state
{
private:
    const packet_memcpy_function orig_memcpy;  ///< Packet memcpy provided by the user
    const chunk_stream_config chunk_config;
    const std::uintptr_t stream_id;
    const std::size_t base_stat_index;         ///< Index of first custom stat
    /// Circular buffer of chunks under construction
    std::vector<std::unique_ptr<chunk>> chunks;
    std::int64_t head_chunk = 0, tail_chunk = 0;  ///< chunk IDs of valid chunk range
    std::size_t head_pos = 0, tail_pos = 0;  ///< Positions corresponding to @ref head and @ref tail in @ref chunks

    void packet_memcpy(const spead2::memory_allocator::pointer &allocation,
                       const packet_header &packet) const;

    /// Send the oldest chunk to the ready callback
    void flush_head();

protected:
    std::int64_t get_head_chunk() const { return head_chunk; }
    std::int64_t get_tail_chunk() const { return tail_chunk; }

public:
    /**
     * Structure associated with each heap, as the deleter of the
     * allocated pointer.
     */
    struct heap_metadata
    {
        std::int64_t chunk_id;
        std::size_t heap_index;
        std::size_t heap_offset;
        chunk *chunk_ptr;

        // Free the memory (no-op since we don't own it)
        void operator()(std::uint8_t *) const {}
    };

    /// Constructor
    chunk_stream_state(const stream_config &config, const chunk_stream_config &chunk_config);

    /// Get the stream's chunk configuration
    const chunk_stream_config &get_chunk_config() const { return chunk_config; }

    /// Compute the config to pass down to @ref spead2::recv::stream.
    stream_config adjust_config(const stream_config &config);

    /**
     * Allocate storage for a heap within a chunk, given a first packet for a
     * heap.
     *
     * @returns A raw pointer for heap storage and context used for actual copies.
     */
    std::pair<std::uint8_t *, heap_metadata> allocate(
        std::size_t size, const packet_header &packet);

    /// Send all in-flight chunks to the ready callback
    void flush_chunks();

    /**
     * Get the @ref heap_metadata associated with a heap payload pointer.
     * If the pointer was not allocated by a chunk stream, returns @c
     * nullptr.
     */
    static const heap_metadata *get_heap_metadata(const memory_allocator::pointer &ptr);
};

/**
 * Custom allocator for a chunk stream.
 *
 * It forwards allocation requests to @ref chunk_stream_state.
 */
class chunk_stream_allocator final : public memory_allocator
{
private:
    chunk_stream_state &stream;

public:
    explicit chunk_stream_allocator(chunk_stream_state &stream);

    virtual pointer allocate(std::size_t size, void *hint) override;
};

} // namespace detail

/**
 * Stream that writes incoming heaps into chunks.
 */
class chunk_stream : private detail::chunk_stream_state, public stream
{
    friend class chunk_stream_state;

    virtual void heap_ready(live_heap &&) override;

public:
    using heap_metadata = detail::chunk_stream_state::heap_metadata;

    /**
     * Constructor.
     *
     * This class passes a modified @a config to the base class constructor.
     * This is reflected in the return of @ref get_config. In particular:
     *
     * - The @link stream_config::set_allow_unsized_heaps allow unsized
     *   heaps@endlink setting is forced to false.
     * - The @link stream_config::set_memcpy memcpy function@endlink may be
     *   overridden, although the provided function is still used when a copy
     *   happens. Use @ref get_heap_metadata to
     *   get a pointer to @ref heap_metadata, from which the chunk can be retrieved.
     * - The @link stream_config::set_memory_allocator memory allocator@endlink
     *   is overridden, and the provided value is ignored.
     * - Additional statistics are registered:
     *   - <tt>too_old_heaps</tt>: number of heaps for which the placement function returned
     *     a non-negative chunk ID that was behind the window.
     *   - <tt>rejected_heaps</tt>: number of heaps for which the placement function returned
     *     a negative chunk ID.
     *
     * @param io_service       I/O service (also used by the readers).
     * @param config           Basic stream configuration
     * @param chunk_config     Configuration for chunking
     *
     * @throw invalid_value if any of the function pointers in @a chunk_config
     * have not been set.
     */
    chunk_stream(
        io_service_ref io_service,
        const stream_config &config,
        const chunk_stream_config &chunk_config);

    using detail::chunk_stream_state::get_chunk_config;
    using detail::chunk_stream_state::get_heap_metadata;

    virtual void stop_received() override;
    virtual void stop() override;
    virtual ~chunk_stream() override;
};

/**
 * Wrapper around @ref chunk_stream that uses ringbuffers to manage chunks.
 *
 * When a fresh chunk is needed, it is retrieved from a ringbuffer of free
 * chunks (the "free ring"). When a chunk is flushed, it is pushed to a
 * "data ring". These may be shared between streams, but both will be
 * stopped as soon as any of the streams using them are stopped. The intended
 * use case is parallel streams that are started and stopped together.
 *
 * When @ref stop is called, any in-flight chunks (that are not in either
 * of the ringbuffers) will be freed from the thread that called @ref stop.
 *
 * It's important to note that the free ring is also stopped if the stream
 * is stopped by a stream control item. The user must thus be prepared to
 * deal gracefully with a @ref ringbuffer_stopped exception when
 * pushing to the free ring.
 */
template<typename DataRingbuffer = ringbuffer<std::unique_ptr<chunk>>,
         typename FreeRingbuffer = ringbuffer<std::unique_ptr<chunk>>>
class chunk_ring_stream : public chunk_stream
{
private:
    std::shared_ptr<DataRingbuffer> data_ring;
    std::shared_ptr<FreeRingbuffer> free_ring;
    /// Temporary storage for in-flight chunks during @ref stop
    std::vector<std::unique_ptr<chunk>> graveyard;

    /// Create a new @ref spead2::recv::chunk_stream_config that uses the ringbuffers
    static chunk_stream_config adjust_chunk_config(
        const chunk_stream_config &chunk_config,
        DataRingbuffer &data_ring,
        FreeRingbuffer &free_ring,
        std::vector<std::unique_ptr<chunk>> &graveyard);

public:
    /**
     * Constructor. Refer to @ref chunk_stream::chunk_stream for details.
     *
     * The @link chunk_stream_config::set_allocate allocate@endlink callback
     * is ignored and should be unset. If a
     * @link chunk_stream_config::set_ready ready@endlink callback is
     * defined, it will be called before the chunk is pushed onto the
     * ringbuffer. It must not move from the provided @a unique_ptr, but it
     * can be used to perform further processing on the chunk before it is
     * pushed.
     *
     * Calling @ref get_chunk_config will reflect the internally-defined
     * callbacks.
     */
    chunk_ring_stream(
        io_service_ref io_service,
        const stream_config &config,
        const chunk_stream_config &chunk_config,
        std::shared_ptr<DataRingbuffer> data_ring,
        std::shared_ptr<FreeRingbuffer> free_ring);

    /**
     * Add a chunk to the free ringbuffer. This takes care of zeroing out
     * the @ref spead2::recv::chunk::present array, and it will suppress the
     * @ref spead2::ringbuffer_stopped error if the free ringbuffer has been
     * stopped (in which case the argument will not have been moved from).
     *
     * If the free ring is full, it will throw @ref spead2::ringbuffer_full
     * rather than blocking. The free ringbuffer should be constructed with
     * enough slots that this does not happen.
     */
    void add_free_chunk(std::unique_ptr<chunk> &&c);

    /// Retrieve the data ringbuffer passed to the constructor
    std::shared_ptr<DataRingbuffer> get_data_ringbuffer() const { return data_ring; }
    /// Retrieve the free ringbuffer passed to the constructor
    std::shared_ptr<FreeRingbuffer> get_free_ringbuffer() const { return free_ring; }

    virtual void stop_received() override;
    virtual void stop() override;
    virtual ~chunk_ring_stream();
};

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::chunk_ring_stream(
        io_service_ref io_service,
        const stream_config &config,
        const chunk_stream_config &chunk_config,
        std::shared_ptr<DataRingbuffer> data_ring,
        std::shared_ptr<FreeRingbuffer> free_ring)
    : chunk_stream(
        io_service,
        config,
        adjust_chunk_config(chunk_config, *data_ring, *free_ring, graveyard)),
    data_ring(std::move(data_ring)),
    free_ring(std::move(free_ring))
{
    this->data_ring->add_producer();
    // Ensure that we don't run out of memory during shutdown
    graveyard.reserve(get_chunk_config().get_max_chunks());
}

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_stream_config chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::adjust_chunk_config(
    const chunk_stream_config &chunk_config,
    DataRingbuffer &data_ring,
    FreeRingbuffer &free_ring,
    std::vector<std::unique_ptr<chunk>> &graveyard)
{
    chunk_stream_config new_config = chunk_config;
    // Set the allocate callback to get a chunk from the free ringbuffer
    new_config.set_allocate([&free_ring] (std::int64_t, std::uint64_t *) -> std::unique_ptr<chunk> {
        try
        {
            return free_ring.pop();
        }
        catch (ringbuffer_stopped &)
        {
            // We're shutting down. Return a null pointer so that we just
            // ignore this chunk
            return nullptr;
        }
    });
    // Set the ready callback to push chunks to the data ringbuffer
    auto orig_ready = chunk_config.get_ready();
    new_config.set_ready(
        [&data_ring, &graveyard, orig_ready] (std::unique_ptr<chunk> &&c,
                                              std::uint64_t *batch_stats) {
        try
        {
            if (orig_ready)
                orig_ready(std::move(c), batch_stats);
            // TODO: use try_push and track stalls
            data_ring.push(std::move(c));
        }
        catch (ringbuffer_stopped &)
        {
            // Suppress the error, move the chunk to the graveyard
            log_info("dropped chunk %d due to external stop", c->chunk_id);
            graveyard.push_back(std::move(c));
        }
    });
    return new_config;
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::add_free_chunk(std::unique_ptr<chunk> &&c)
{
    // Mark all heaps as not yet present
    std::memset(c->present.get(), 0, c->present_size);
    try
    {
        free_ring->try_push(std::move(c));
    }
    catch (spead2::ringbuffer_stopped &)
    {
        // Suppress the error
    }
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::stop_received()
{
    chunk_stream::stop_received();
    data_ring->remove_producer();
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::stop()
{
    // Stop the ringbuffers first, so that if the calling code is no longer
    // servicing them it will not lead to a deadlock as we flush.
    free_ring->stop();
    data_ring->stop();  // NB: NOT remove_producer as that might not break a deadlock
    chunk_stream::stop();
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        graveyard.clear(); // free chunks that didn't make it into data_ring
    }
}

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::~chunk_ring_stream()
{
    // Flush before the references to the rings get lost
    stop();
}

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_CHUNK_STREAM
