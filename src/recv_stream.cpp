/* Copyright 2015, 2017-2020 National Research Foundation (SARAO)
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

#include <cstddef>
#include <utility>
#include <algorithm>
#include <cassert>
#include <atomic>
#include <spead2/recv_stream.h>
#include <spead2/recv_live_heap.h>
#include <spead2/common_memcpy.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_logging.h>

#define INVALID_ENTRY ((queue_entry *) -1)

namespace spead2
{
namespace recv
{

stream_stats stream_stats::operator+(const stream_stats &other) const
{
    stream_stats out = *this;
    out += other;
    return out;
}

stream_stats &stream_stats::operator+=(const stream_stats &other)
{
    heaps += other.heaps;
    incomplete_heaps_evicted += other.incomplete_heaps_evicted;
    incomplete_heaps_flushed += other.incomplete_heaps_flushed;
    packets += other.packets;
    batches += other.batches;
    worker_blocked += other.worker_blocked;
    max_batch = std::max(max_batch, other.max_batch);
    single_packet_heaps += other.single_packet_heaps;
    search_dist += other.search_dist;
    return *this;
}

constexpr std::size_t stream_config::default_max_heaps;

static std::size_t compute_bucket_count(std::size_t max_heaps)
{
    std::size_t buckets = 4;
    while (buckets < max_heaps)
        buckets *= 2;
    buckets *= 4;    // Make sure the table has a low load factor
    return buckets;
}

/* Compute shift such that (x >> shift) < bucket_count for all 64-bit x
 * and bucket_count a power of 2.
 */
static int compute_bucket_shift(std::size_t bucket_count)
{
    int shift = 64;
    while (bucket_count > 1)
    {
        shift--;
        bucket_count >>= 1;
    }
    return shift;
}

static void packet_memcpy_std(const spead2::memory_allocator::pointer &allocation, const packet_header &packet)
{
    std::memcpy(allocation.get() + packet.payload_offset, packet.payload, packet.payload_length);
}

static void packet_memcpy_nontemporal(const spead2::memory_allocator::pointer &allocation, const packet_header &packet)
{
    spead2::memcpy_nontemporal(allocation.get() + packet.payload_offset, packet.payload, packet.payload_length);
}

stream_config::stream_config()
    : memcpy(packet_memcpy_std),
    allocator(std::make_shared<memory_allocator>())
{
}

stream_config &stream_config::set_max_heaps(std::size_t max_heaps)
{
    if (max_heaps == 0)
        throw std::invalid_argument("max_heaps cannot be 0");
    this->max_heaps = max_heaps;
    return *this;
}

stream_config &stream_config::set_bug_compat(bug_compat_mask bug_compat)
{
    if (bug_compat & ~BUG_COMPAT_PYSPEAD_0_5_2)
        throw std::invalid_argument("unknown compatibility bits in bug_compat");
    this->bug_compat = bug_compat;
    return *this;
}

stream_config &stream_config::set_memory_allocator(std::shared_ptr<memory_allocator> allocator)
{
    this->allocator = std::move(allocator);
    return *this;
}

stream_config &stream_config::set_memcpy(packet_memcpy_function memcpy)
{
    this->memcpy = memcpy;
    return *this;
}

stream_config &stream_config::set_memcpy(memcpy_function memcpy)
{
    return set_memcpy(
        packet_memcpy_function([memcpy](
            const spead2::memory_allocator::pointer &allocation, const packet_header &packet)
        {
            memcpy(allocation.get() + packet.payload_offset, packet.payload, packet.payload_length);
        })
    );
}

stream_config &stream_config::set_memcpy(memcpy_function_id id)
{
    /* We adapt each case to the packet_memcpy signature rather than using the
     * generic wrapping in the memcpy_function overload. This ensures that
     * there is only one level of indirect function call instead of two. It
     * also makes it possible to reverse the mapping by comparing function
     * pointers.
     */
    switch (id)
    {
    case MEMCPY_STD:
        set_memcpy(packet_memcpy_std);
        break;
    case MEMCPY_NONTEMPORAL:
        set_memcpy(packet_memcpy_nontemporal);
        break;
    default:
        throw std::invalid_argument("Unknown memcpy function");
    }
    return *this;
}

stream_config &stream_config::set_stop_on_stop_item(bool stop)
{
    stop_on_stop_item = stop;
    return *this;
}

stream_config &stream_config::set_allow_unsized_heaps(bool allow)
{
    allow_unsized_heaps = allow;
    return *this;
}

stream_config &stream_config::set_allow_out_of_order(bool allow)
{
    allow_out_of_order = allow;
    return *this;
}

stream_base::stream_base(const stream_config &config)
    : queue_storage(new storage_type[config.get_max_heaps()]),
    bucket_count(compute_bucket_count(config.get_max_heaps())),
    bucket_shift(compute_bucket_shift(bucket_count)),
    buckets(new queue_entry *[bucket_count]),
    head(0),
    config(config)
{
    for (std::size_t i = 0; i < config.get_max_heaps(); i++)
        cast(i)->next = INVALID_ENTRY;
    for (std::size_t i = 0; i < bucket_count; i++)
        buckets[i] = NULL;
}

stream_base::~stream_base()
{
    for (std::size_t i = 0; i < get_config().get_max_heaps(); i++)
    {
        queue_entry *entry = cast(i);
        if (entry->next != INVALID_ENTRY)
        {
            unlink_entry(entry);
            entry->heap.~live_heap();
        }
    }
}

std::size_t stream_base::get_bucket(s_item_pointer_t heap_cnt) const
{
    // Look up Fibonacci hashing for an explanation of the magic number
    return (heap_cnt * 11400714819323198485ULL) >> bucket_shift;
}

stream_base::queue_entry *stream_base::cast(std::size_t index)
{
    return reinterpret_cast<queue_entry *>(&queue_storage[index]);
}

void stream_base::unlink_entry(queue_entry *entry)
{
    assert(entry->next != INVALID_ENTRY);
    std::size_t bucket_id = get_bucket(entry->heap.get_cnt());
    queue_entry **prev = &buckets[bucket_id];
    while (*prev != entry)
    {
        assert(*prev != NULL && *prev != INVALID_ENTRY);
        prev = &(*prev)->next;
    }
    *prev = entry->next;
    entry->next = INVALID_ENTRY;
}

stream_base::add_packet_state::add_packet_state(stream_base &owner)
    : owner(owner), lock(owner.queue_mutex)
{
}

stream_base::add_packet_state::~add_packet_state()
{
    std::lock_guard<std::mutex> stats_lock(owner.stats_mutex);
    if (!packets && is_stopped())
        return;   // Stream was stopped before we could do anything - don't count as a batch
    owner.stats.packets += packets;
    owner.stats.batches++;
    owner.stats.heaps += complete_heaps + incomplete_heaps_evicted;
    owner.stats.incomplete_heaps_evicted += incomplete_heaps_evicted;
    owner.stats.single_packet_heaps += single_packet_heaps;
    owner.stats.search_dist += search_dist;
    owner.stats.max_batch = std::max(owner.stats.max_batch, std::size_t(packets));
}

bool stream_base::add_packet(add_packet_state &state, const packet_header &packet)
{
    const stream_config &config = state.owner.get_config();
    assert(!stopped);
    state.packets++;
    if (packet.heap_length < 0 && !config.get_allow_unsized_heaps())
    {
        log_info("packet rejected because it has no HEAP_LEN");
        return false;
    }

    // Look for matching heap.
    queue_entry *entry = NULL;
    s_item_pointer_t heap_cnt = packet.heap_cnt;
    std::size_t bucket_id = get_bucket(heap_cnt);
    assert(bucket_id < bucket_count);
    if (packet.heap_length >= 0 && packet.payload_length == packet.heap_length)
    {
        // Packet is a complete heap, so it shouldn't match any partial heap.
        entry = NULL;
        state.single_packet_heaps++;
    }
    else
    {
        int search_dist = 1;
        for (entry = buckets[bucket_id]; entry != NULL; entry = entry->next, search_dist++)
        {
            assert(entry != INVALID_ENTRY);
            if (entry->heap.get_cnt() == heap_cnt)
                break;
        }
        state.search_dist += search_dist;
    }

    if (!entry)
    {
        /* Never seen this heap before. Evict the old one in its slot,
         * if any. However, if we're in in-order mode, only accept the
         * packet if it is supposed to be at the start of the heap.
         *
         * Note: not safe to dereference h just anywhere here!
         */
        if (!config.get_allow_out_of_order() && packet.payload_offset != 0)
        {
            log_debug("packet rejected because there is a gap in the heap and "
                      "allow_out_of_order is false");
            return false;
        }

        if (++head == config.get_max_heaps())
            head = 0;
        entry = cast(head);
        if (entry->next != INVALID_ENTRY)
        {
            state.incomplete_heaps_evicted++;
            unlink_entry(entry);
            heap_ready(std::move(entry->heap));
            entry->heap.~live_heap();
        }
        entry->next = buckets[bucket_id];
        buckets[bucket_id] = entry;
        new (&entry->heap) live_heap(packet, config.get_bug_compat());
    }

    live_heap *h = &entry->heap;
    bool result = false;
    bool end_of_stream = false;
    if (h->add_packet(packet, config.get_memcpy(), *config.get_memory_allocator(),
                      config.get_allow_out_of_order()))
    {
        result = true;
        end_of_stream = config.get_stop_on_stop_item() && h->is_end_of_stream();
        if (h->is_complete())
        {
            unlink_entry(entry);
            if (!end_of_stream)
            {
                state.complete_heaps++;
                heap_ready(std::move(*h));
            }
            h->~live_heap();
        }
    }

    if (end_of_stream)
        stop_received();
    return result;
}

void stream_base::flush_unlocked()
{
    const std::size_t max_heaps = get_config().get_max_heaps();
    std::size_t n_flushed = 0;
    for (std::size_t i = 0; i < max_heaps; i++)
    {
        if (++head == max_heaps)
            head = 0;
        queue_entry *entry = cast(head);
        if (entry->next != INVALID_ENTRY)
        {
            n_flushed++;
            unlink_entry(entry);
            heap_ready(std::move(entry->heap));
            entry->heap.~live_heap();
        }
    }
    std::lock_guard<std::mutex> stats_lock(stats_mutex);
    stats.heaps += n_flushed;
    stats.incomplete_heaps_flushed += n_flushed;
}

void stream_base::flush()
{
    std::lock_guard<std::mutex> lock(queue_mutex);
    flush_unlocked();
}

void stream_base::stop_unlocked()
{
    if (!stopped)
        stop_received();
}

void stream_base::stop()
{
    std::lock_guard<std::mutex> lock(queue_mutex);
    stop_unlocked();
}

void stream_base::stop_received()
{
    assert(!stopped);
    stopped = true;
    flush_unlocked();
}

stream_stats stream_base::get_stats() const
{
    std::lock_guard<std::mutex> stats_lock(stats_mutex);
    stream_stats ret = stats;
    return ret;
}


stream::stream(io_service_ref io_service, const stream_config &config)
    : stream_base(config),
    thread_pool_holder(std::move(io_service).get_shared_thread_pool()),
    io_service(*io_service)
{
}

void stream::stop_received()
{
    stream_base::stop_received();
    std::lock_guard<std::mutex> lock(reader_mutex);
    for (const auto &reader : readers)
        reader->stop();
}

void stream::stop_impl()
{
    stream_base::stop();

    std::size_t n_readers;
    {
        std::lock_guard<std::mutex> lock(reader_mutex);
        /* Prevent any further calls to emplace_reader from doing anything, so
         * that n_readers will remain accurate.
         */
        stop_readers = true;
        n_readers = readers.size();
    }

    // Wait until all readers have wound up all their completion handlers
    while (n_readers > 0)
    {
        semaphore_get(readers_stopped);
        n_readers--;
    }

    {
        /* This lock is not strictly needed since no other thread can touch
         * readers any more, but is harmless.
         */
        std::lock_guard<std::mutex> lock(reader_mutex);
        readers.clear();
    }
}

void stream::stop()
{
    std::call_once(stop_once, [this] { stop_impl(); });
}

bool stream::is_lossy() const
{
    std::lock_guard<std::mutex> lock(reader_mutex);
    return lossy;
}

stream::~stream()
{
    stop();
}


const std::uint8_t *mem_to_stream(stream_base &s, const std::uint8_t *ptr, std::size_t length)
{
    stream_base::add_packet_state state(s);
    while (length > 0 && !state.is_stopped())
    {
        packet_header packet;
        std::size_t size = decode_packet(packet, ptr, length);
        if (size > 0)
        {
            state.add_packet(packet);
            ptr += size;
            length -= size;
        }
        else
            length = 0; // causes loop to exit
    }
    return ptr;
}

} // namespace recv
} // namespace spead2
