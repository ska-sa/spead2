/* Copyright 2021-2023 National Research Foundation (SARAO)
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

#include <cassert>
#include <memory>
#include <vector>
#include <functional>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <mutex>
#include <limits>
#include <spead2/common_defines.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_endian.h>
#include <spead2/recv_packet.h>
#include <spead2/recv_stream.h>

namespace spead2::recv
{

namespace detail
{

template<typename DataRingbuffer, typename FreeRingbuffer> class chunk_ring_pair;

} // namespace detail

/// Storage for a chunk with metadata
class chunk
{
    friend class chunk_stream_group;
    template<typename DataRingbuffer, typename FreeRingbuffer> friend class detail::chunk_ring_pair;
private:
    /// Linked list of chunks to dispose of at shutdown
    std::unique_ptr<chunk> graveyard_next;

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
    /// Optional storage area for per-heap metadata
    memory_allocator::pointer extra;

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
 *
 * Do not modify any of the pointers in the structure.
 */
struct chunk_place_data
{
    const std::uint8_t *packet;      ///< Pointer to the original packet data
    std::size_t packet_size;         ///< Number of bytes referenced by @ref packet
    s_item_pointer_t *items;         ///< Values of requested item pointers
    /// Chunk ID (output). Set to -1 (or leave unmodified) to discard the heap.
    std::int64_t chunk_id;
    std::size_t heap_index;          ///< Number of this heap within the chunk (output)
    std::size_t heap_offset;         ///< Byte offset of this heap within the chunk payload (output)
    std::uint64_t *batch_stats;      ///< Pointer to staging area for statistics
    std::uint8_t *extra;             ///< Pointer to data to be copied to @ref chunk::extra
    std::size_t extra_offset;        ///< Offset within @ref chunk::extra to write
    std::size_t extra_size;          ///< Number of bytes to copy to @ref chunk::extra
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
    std::size_t max_heap_extra = 0;

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

    /// Set maximum amount of data a placement function may write to @ref chunk_place_data::extra.
    chunk_stream_config &set_max_heap_extra(std::size_t max_heap_extra);
    /// Get maximum amount of data a placement function may write to @ref chunk_place_data::extra.
    std::size_t get_max_heap_extra() const { return max_heap_extra; }
};

namespace detail
{

/**
 * Sliding window of chunk pointers.
 *
 * @internal The chunk IDs are kept as unsigned values, so that the tail can
 * be larger than any actual chunk ID.
 */
class chunk_window
{
private:
    /// Circular buffer of chunks under construction.
    std::vector<chunk *> chunks;
    std::uint64_t head_chunk = 0, tail_chunk = 0;  ///< chunk IDs of valid chunk range
    std::size_t head_pos = 0, tail_pos = 0;  ///< Positions corresponding to @ref head and @ref tail in @ref chunks

public:
    /// Send the oldest chunk to the ready callback
    template<typename F>
    void flush_head(const F &ready_chunk)
    {
        assert(head_chunk < tail_chunk);
        if (chunks[head_pos])
        {
            ready_chunk(chunks[head_pos]);
            chunks[head_pos] = nullptr;
        }
        head_chunk++;
        head_pos++;
        if (head_pos == chunks.size())
            head_pos = 0;  // wrap around the circular buffer
    }

    /// Send the oldest chunk to the ready callback
    template<typename F1, typename F2>
    void flush_head(const F1 &ready_chunk, const F2 &head_updated)
    {
        flush_head(ready_chunk);
        head_updated(head_chunk);
    }

    /**
     * Send all the chunks to the ready callback. Afterwards,
     * the head and tail are both advanced to @a next_chunk.
     */
    template<typename F1, typename F2>
    void flush_all(std::uint64_t next_chunk, const F1 &ready_chunk, const F2 &head_updated)
    {
        std::uint64_t orig_head = head_chunk;
        while (!empty())
            flush_head(ready_chunk);
        head_chunk = tail_chunk = next_chunk;
        if (head_chunk != orig_head)
            head_updated(head_chunk);
    }

    /// Flush until the head is at least @a target
    template<typename F1, typename F2>
    void flush_until(std::uint64_t target, const F1 &ready_chunk, const F2 &head_updated)
    {
        if (head_chunk < target)
        {
            while (head_chunk != tail_chunk && head_chunk < target)
                flush_head(ready_chunk);
            if (head_chunk < target)
                head_chunk = tail_chunk = target;
            head_updated(target);
        }
    }

    explicit chunk_window(std::size_t max_chunks);

    /**
     * Obtain a pointer to a chunk with ID @a chunk_id.
     *
     * If @a chunk_id falls outside the window, returns nullptr.
     */
    chunk *get_chunk(std::uint64_t chunk_id) const
    {
        if (chunk_id >= head_chunk && chunk_id < tail_chunk)
        {
            std::size_t pos = chunk_id - head_chunk + head_pos;
            const std::size_t max_chunks = chunks.size();
            if (pos >= max_chunks)
                pos -= max_chunks;  // wrap around the circular storage
            return chunks[pos];
        }
        else
            return nullptr;
    }

    /**
     * Obtain a pointer to a chunk with ID @a chunk_id.
     *
     * If @a chunk_id is behind the window, returns nullptr. If it is ahead of
     * the window, the window is advanced using @a allocate_chunk and
     * @a ready_chunk. If the head_chunk is updated, the new value is passed to
     * @a head_updated.
     */
    template<typename F1, typename F2, typename F3>
    chunk *get_chunk(
        std::uint64_t chunk_id, std::uintptr_t stream_id,
        const F1 &allocate_chunk, const F2 &ready_chunk, const F3 &head_updated)
    {
        // chunk_id must be a valid int64_t
        assert(chunk_id <= std::uint64_t(std::numeric_limits<std::int64_t>::max()));
        const std::size_t max_chunks = chunks.size();
        if (chunk_id >= head_chunk)
        {
            // We've moved beyond the end of our current window, and need to
            // allocate fresh chunks.
            if (chunk_id >= tail_chunk && chunk_id - tail_chunk >= max_chunks)
            {
                /* We've jumped ahead so far that the entire current window
                 * is stale. Flush it all and fast-forward to the new window.
                 * We leave it to the while loop below to actually allocate
                 * the chunks.
                 */
                flush_all(chunk_id - (max_chunks - 1), ready_chunk, head_updated);
            }
            while (chunk_id >= tail_chunk)
            {
                if (tail_chunk - head_chunk == max_chunks)
                    flush_head(ready_chunk, head_updated);
                chunks[tail_pos] = allocate_chunk(tail_chunk);
                if (chunks[tail_pos])
                {
                    chunks[tail_pos]->chunk_id = tail_chunk;
                    chunks[tail_pos]->stream_id = stream_id;
                }
                tail_chunk++;
                tail_pos++;
                if (tail_pos == max_chunks)
                    tail_pos = 0;  // wrap around circular buffer
            }
            // Find position of chunk within the storage
            std::size_t pos = chunk_id - head_chunk + head_pos;
            if (pos >= max_chunks)
                pos -= max_chunks;  // wrap around the circular storage
            return chunks[pos];
        }
        else
            return nullptr;
    }

    std::uint64_t get_head_chunk() const { return head_chunk; }
    std::uint64_t get_tail_chunk() const { return tail_chunk; }
    bool empty() const { return head_chunk == tail_chunk; }
};

template<typename CM> class chunk_stream_allocator;

/// Parts of @ref chunk_stream_state that don't depend on the chunk manager
class chunk_stream_state_base
{
protected:
    struct free_place_data
    {
        void operator()(unsigned char *ptr) const;
    };

    const packet_memcpy_function orig_memcpy;  ///< Packet memcpy provided by the user
    const chunk_stream_config chunk_config;
    const std::uintptr_t stream_id;
    const std::size_t base_stat_index;         ///< Index of first custom stat

    /**
     * Circular buffer of chunks under construction.
     *
     * This class might or might not have exclusive ownership of the chunks,
     * depending on the template parameter.
     */
    chunk_window chunks;

    /**
     * Scratch area for use by @ref chunk_place_function. This contains not
     * just the @ref chunk_place_data, but also the various arrays it points
     * to. They're allocated contiguously to minimise the number of cache lines
     * accessed.
     */
    std::unique_ptr<unsigned char[], free_place_data> place_data_storage;
    chunk_place_data *place_data;

    void packet_memcpy(const spead2::memory_allocator::pointer &allocation,
                       const packet_header &packet) const;

    /// Implementation of @ref stream::heap_ready
    void do_heap_ready(live_heap &&lh);

protected:
    std::uint64_t get_head_chunk() const { return chunks.get_head_chunk(); }
    std::uint64_t get_tail_chunk() const { return chunks.get_tail_chunk(); }
    bool chunk_too_old(std::int64_t chunk_id) const
    {
        // Need to check against 0 explicitly to avoid signed/unsigned mixup
        return chunk_id < 0 || std::uint64_t(chunk_id) < chunks.get_head_chunk();
    }

public:
    /// Constructor
    chunk_stream_state_base(
        const stream_config &config,
        const chunk_stream_config &chunk_config);

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

    /// Get the stream's chunk configuration
    const chunk_stream_config &get_chunk_config() const { return chunk_config; }

    /**
     * Get the metadata associated with a heap payload pointer.  If the pointer
     * was not allocated by a chunk stream, returns @c nullptr.
     */
    static const heap_metadata *get_heap_metadata(const memory_allocator::pointer &ptr);
};

/**
 * Base class that holds the internal state of @ref
 * spead2::recv::chunk_stream.
 *
 * This is split into a separate class to avoid some initialisation ordering
 * problems: it is constructed before the @ref spead2::recv::stream base class,
 * allowing the latter to use function objects that reference this class.
 *
 * The template parameter allows the policy for allocating and releasing
 * chunks to be customised. See @ref chunk_manager_simple for the API.
 */
template<typename CM>
class chunk_stream_state : public chunk_stream_state_base
{
private:
    using chunk_manager_t = CM;
    friend chunk_manager_t;

    chunk_manager_t chunk_manager;

public:
    /// Constructor
    chunk_stream_state(
        const stream_config &config,
        const chunk_stream_config &chunk_config,
        chunk_manager_t chunk_manager);

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

    /// Send all in-flight chunks to the ready callback (not thread-safe)
    void flush_chunks();
};

class chunk_manager_simple
{
public:
    explicit chunk_manager_simple(const chunk_stream_config &chunk_config);

    std::uint64_t *get_batch_stats(chunk_stream_state<chunk_manager_simple> &state) const;
    chunk *allocate_chunk(chunk_stream_state<chunk_manager_simple> &state, std::int64_t chunk_id);
    void ready_chunk(chunk_stream_state<chunk_manager_simple> &state, chunk *c);
    void head_updated(chunk_stream_state<chunk_manager_simple> &, std::uint64_t) {}
};

/**
 * Custom allocator for a chunk stream.
 *
 * It forwards allocation requests to @ref chunk_stream_state.
 */
template<typename CM>
class chunk_stream_allocator final : public memory_allocator
{
private:
    chunk_stream_state<CM> &stream;

public:
    explicit chunk_stream_allocator(chunk_stream_state<CM> &stream);

    virtual pointer allocate(std::size_t size, void *hint) override;
};

template<typename CM>
chunk_stream_allocator<CM>::chunk_stream_allocator(chunk_stream_state<CM> &stream)
    : stream(stream)
{
}

template<typename CM>
memory_allocator::pointer chunk_stream_allocator<CM>::allocate(std::size_t size, void *hint)
{
    if (hint)
    {
        auto [ptr, metadata] = stream.allocate(size, *reinterpret_cast<const packet_header *>(hint));
        // Use the heap_metadata as the deleter
        return pointer(ptr, std::move(metadata));
    }
    // Probably unreachable, but provides a safety net
    return memory_allocator::allocate(size, hint);
}

extern template class chunk_stream_state<chunk_manager_simple>;
extern template class chunk_stream_allocator<chunk_manager_simple>;

} // namespace detail

/**
 * Stream that writes incoming heaps into chunks.
 */
class chunk_stream : private detail::chunk_stream_state<detail::chunk_manager_simple>, public stream
{
    friend class detail::chunk_stream_state<detail::chunk_manager_simple>;
    friend class detail::chunk_manager_simple;

    virtual void heap_ready(live_heap &&) override;

public:
    using heap_metadata = detail::chunk_stream_state_base::heap_metadata;

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
     * @throw invalid_argument if any of the function pointers in @a chunk_config
     * have not been set.
     */
    chunk_stream(
        io_service_ref io_service,
        const stream_config &config,
        const chunk_stream_config &chunk_config);

    using detail::chunk_stream_state_base::get_chunk_config;
    using detail::chunk_stream_state_base::get_heap_metadata;

    virtual void stop_received() override;
    virtual void stop() override;
    virtual ~chunk_stream() override;
};

namespace detail
{

/// Common functionality between @ref chunk_ring_stream and @ref chunk_stream_ring_group
template<typename DataRingbuffer = ringbuffer<std::unique_ptr<chunk>>,
         typename FreeRingbuffer = ringbuffer<std::unique_ptr<chunk>>>
class chunk_ring_pair
{
protected:
    const std::shared_ptr<DataRingbuffer> data_ring;
    const std::shared_ptr<FreeRingbuffer> free_ring;
    /// Temporary stroage for linked list of in-flight chunks while stopping
    std::unique_ptr<chunk> graveyard;

    chunk_ring_pair(std::shared_ptr<DataRingbuffer> data_ring, std::shared_ptr<FreeRingbuffer> free_ring);

public:
    /// Create an allocate function that obtains chunks from the free ring
    chunk_allocate_function make_allocate();
    /**
     * Create a ready function that pushes chunks to the data ring.
     *
     * The orig_ready function is called first.
     */
    chunk_ready_function make_ready(const chunk_ready_function &orig_ready);

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
};

} // namespace detail

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
 */
template<typename DataRingbuffer = ringbuffer<std::unique_ptr<chunk>>,
         typename FreeRingbuffer = ringbuffer<std::unique_ptr<chunk>>>
class chunk_ring_stream : public detail::chunk_ring_pair<DataRingbuffer, FreeRingbuffer>, public chunk_stream
{
private:
    /// Create a new @ref spead2::recv::chunk_stream_config that uses the ringbuffers
    static chunk_stream_config adjust_chunk_config(
        const chunk_stream_config &chunk_config,
        detail::chunk_ring_pair<DataRingbuffer, FreeRingbuffer> &ring_pair);

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

    virtual void stop_received() override;
    virtual void stop() override;
    virtual ~chunk_ring_stream();
};

namespace detail
{

template<typename CM>
chunk_stream_state<CM>::chunk_stream_state(
    const stream_config &config,
    const chunk_stream_config &chunk_config,
    chunk_manager_t chunk_manager)
    : chunk_stream_state_base(config, chunk_config),
    chunk_manager(std::move(chunk_manager))
{
}

template<typename CM>
stream_config chunk_stream_state<CM>::adjust_config(const stream_config &config)
{
    using namespace std::placeholders;
    stream_config new_config = config;
    // Unsized heaps won't work with the custom allocator
    new_config.set_allow_unsized_heaps(false);
    new_config.set_memory_allocator(std::make_shared<chunk_stream_allocator<chunk_manager_t>>(*this));
    // Override the original memcpy with our custom version
    new_config.set_memcpy(std::bind(&chunk_stream_state::packet_memcpy, this, _1, _2));
    // Add custom statistics
    new_config.add_stat("too_old_heaps");
    new_config.add_stat("rejected_heaps");
    return new_config;
}

template<typename CM>
void chunk_stream_state<CM>::flush_chunks()
{
    chunks.flush_all(
        std::numeric_limits<std::uint64_t>::max(),
        [this](chunk *c) { chunk_manager.ready_chunk(*this, c); },
        [this](std::uint64_t head_chunk) { chunk_manager.head_updated(*this, head_chunk); }
    );
}

template<typename CM>
std::pair<std::uint8_t *, chunk_stream_state_base::heap_metadata>
chunk_stream_state<CM>::allocate(std::size_t /* size */, const packet_header &packet)
{
    // Used to get a non-null pointer
    static std::uint8_t dummy_uint8;

    // Keep these in sync with stats added in adjust_config
    static constexpr std::size_t too_old_heaps_offset = 0;
    static constexpr std::size_t rejected_heaps_offset = 1;

    /* Extract the user's requested items.
     * TODO: this could possibly be optimised with a hash table (with a
     * perfect hash function chosen in advance), but for the expected
     * sizes the overheads will probably outweight the benefits.
     */
    const auto &item_ids = get_chunk_config().get_items();
    std::fill(place_data->items, place_data->items + item_ids.size(), -1);
    pointer_decoder decoder(packet.heap_address_bits);
    /* packet.pointers and packet.n_items skips initial "special" item
     * pointers. To allow them to be matched as well, we start from the
     * original packet and skip over the 8-byte header.
     */
    for (const std::uint8_t *p = packet.packet + 8; p != packet.payload; p += sizeof(item_pointer_t))
    {
        item_pointer_t pointer = load_be<item_pointer_t>(p);
        if (decoder.is_immediate(pointer))
        {
            item_pointer_t id = decoder.get_id(pointer);
            for (std::size_t j = 0; j < item_ids.size(); j++)
                if (item_ids[j] == id)
                    place_data->items[j] = decoder.get_immediate(pointer);
        }
    }

    /* TODO: see if the storage can be in the class with the deleter
     * just referencing it. That will avoid the implied memory allocation
     * in constructing the std::function underlying the deleter.
     */
    std::pair<std::uint8_t *, heap_metadata> out;
    auto &[ptr, metadata] = out;
    ptr = &dummy_uint8;  // Use a non-null value to avoid confusion with empty pointers

    place_data->packet = packet.packet;
    place_data->packet_size = packet.payload + packet.payload_length - packet.packet;
    place_data->chunk_id = -1;
    place_data->heap_index = 0;
    place_data->heap_offset = 0;
    place_data->batch_stats = chunk_manager.get_batch_stats(*this);
    place_data->extra_offset = 0;
    place_data->extra_size = 0;
    chunk_config.get_place()(place_data, sizeof(*place_data));
    std::int64_t chunk_id = place_data->chunk_id;
    if (chunk_too_old(chunk_id))
    {
        // We don't want this heap.
        metadata.chunk_id = -1;
        metadata.chunk_ptr = nullptr;
        std::size_t stat_offset = (chunk_id >= 0) ? too_old_heaps_offset : rejected_heaps_offset;
        place_data->batch_stats[base_stat_index + stat_offset]++;
        return out;
    }
    else
    {
        chunk *chunk_ptr = chunks.get_chunk(
            chunk_id,
            stream_id,
            [this](std::int64_t chunk_id) { return chunk_manager.allocate_chunk(*this, chunk_id); },
            [this](chunk *c) { chunk_manager.ready_chunk(*this, c); },
            [this](std::uint64_t head_chunk) { chunk_manager.head_updated(*this, head_chunk); }
        );
        if (chunk_ptr)
        {
            chunk &c = *chunk_ptr;
            ptr = c.data.get() + place_data->heap_offset;
            metadata.chunk_id = chunk_id;
            metadata.heap_index = place_data->heap_index;
            metadata.heap_offset = place_data->heap_offset;
            metadata.chunk_ptr = &c;
            if (place_data->extra_size > 0)
            {
                assert(place_data->extra_size <= chunk_config.get_max_heap_extra());
                assert(c.extra);
                std::memcpy(c.extra.get() + place_data->extra_offset, place_data->extra, place_data->extra_size);
            }
            return out;
        }
        else
        {
            // the allocator didn't allocate a chunk for this slot.
            metadata.chunk_id = -1;
            metadata.chunk_ptr = nullptr;
            return out;
        }
    }
}

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_ring_pair<DataRingbuffer, FreeRingbuffer>::chunk_ring_pair(
    std::shared_ptr<DataRingbuffer> data_ring,
    std::shared_ptr<FreeRingbuffer> free_ring)
    : data_ring(std::move(data_ring)), free_ring(std::move(free_ring))
{
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_ring_pair<DataRingbuffer, FreeRingbuffer>::add_free_chunk(std::unique_ptr<chunk> &&c)
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
chunk_allocate_function chunk_ring_pair<DataRingbuffer, FreeRingbuffer>::make_allocate()
{
    FreeRingbuffer &free_ring = *this->free_ring;
    return [&free_ring] (std::int64_t, std::uint64_t *) -> std::unique_ptr<chunk> {
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
    };
}

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_ready_function chunk_ring_pair<DataRingbuffer, FreeRingbuffer>::make_ready(
    const chunk_ready_function &orig_ready)
{
    DataRingbuffer &data_ring = *this->data_ring;
    std::unique_ptr<chunk> &graveyard = this->graveyard;
    return [&data_ring, &graveyard, orig_ready] (std::unique_ptr<chunk> &&c,
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
            assert(!c->graveyard_next);  // chunk should not already be in a linked list
            c->graveyard_next = std::move(graveyard);
            graveyard = std::move(c);
        }
    };
}

} // namespace detail

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::chunk_ring_stream(
        io_service_ref io_service,
        const stream_config &config,
        const chunk_stream_config &chunk_config,
        std::shared_ptr<DataRingbuffer> data_ring,
        std::shared_ptr<FreeRingbuffer> free_ring)
    : detail::chunk_ring_pair<DataRingbuffer, FreeRingbuffer>(std::move(data_ring), std::move(free_ring)),
    chunk_stream(
        io_service,
        config,
        adjust_chunk_config(chunk_config, *this))
{
    this->data_ring->add_producer();
}

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_stream_config chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::adjust_chunk_config(
    const chunk_stream_config &chunk_config,
    detail::chunk_ring_pair<DataRingbuffer, FreeRingbuffer> &ring_pair)
{
    chunk_stream_config new_config = chunk_config;
    // Set the allocate callback to get a chunk from the free ringbuffer
    new_config.set_allocate(ring_pair.make_allocate());
    // Set the ready callback to push chunks to the data ringbuffer
    auto orig_ready = chunk_config.get_ready();
    new_config.set_ready(ring_pair.make_ready(chunk_config.get_ready()));
    return new_config;
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::stop_received()
{
    chunk_stream::stop_received();
    this->data_ring->remove_producer();
}

template<typename DataRingbuffer, typename FreeRingbuffer>
void chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::stop()
{
    // Stop the ringbuffers first, so that if the calling code is no longer
    // servicing them it will not lead to a deadlock as we flush.
    this->free_ring->stop();
    this->data_ring->stop();  // NB: NOT remove_producer as that might not break a deadlock
    chunk_stream::stop();
    {
        // Locking is probably not needed, as all readers are terminated by
        // chunk_stream::stop(). But it should be safe.
        std::lock_guard<std::mutex> lock(get_queue_mutex());
        this->graveyard.reset(); // free chunks that didn't make it into data_ring
    }
}

template<typename DataRingbuffer, typename FreeRingbuffer>
chunk_ring_stream<DataRingbuffer, FreeRingbuffer>::~chunk_ring_stream()
{
    // Flush before the references to the rings get lost
    stop();
}

} // namespace spead2::recv

#endif // SPEAD2_RECV_CHUNK_STREAM
