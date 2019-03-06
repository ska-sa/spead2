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

constexpr std::size_t stream_base::default_max_heaps;

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

#define SPEAD2_ADAPT_MEMCPY(func, capture) \
    (packet_memcpy_function([capture](const spead2::memory_allocator::pointer &allocation, const packet_header &packet) \
     { \
         func(allocation.get() + packet.payload_offset, packet.payload, packet.payload_length); \
     }))

stream_base::stream_base(bug_compat_mask bug_compat, std::size_t max_heaps)
    : queue_storage(new storage_type[max_heaps]),
    bucket_count(compute_bucket_count(max_heaps)),
    bucket_shift(compute_bucket_shift(bucket_count)),
    buckets(new queue_entry *[bucket_count]),
    head(0),
    max_heaps(max_heaps), bug_compat(bug_compat),
    memcpy(SPEAD2_ADAPT_MEMCPY(std::memcpy, )),
    allocator(std::make_shared<memory_allocator>())
{
    if (max_heaps == 0)
        throw std::invalid_argument("max_heaps cannot be 0");
    for (std::size_t i = 0; i < max_heaps; i++)
        cast(i)->next = INVALID_ENTRY;
    for (std::size_t i = 0; i < bucket_count; i++)
        buckets[i] = NULL;
}

stream_base::~stream_base()
{
    for (std::size_t i = 0; i < max_heaps; i++)
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

void stream_base::set_memory_pool(std::shared_ptr<memory_pool> pool)
{
    set_memory_allocator(std::move(pool));
}

void stream_base::set_memory_allocator(std::shared_ptr<memory_allocator> allocator)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    this->allocator = std::move(allocator);
}

void stream_base::set_memcpy(packet_memcpy_function memcpy)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    this->memcpy = memcpy;
}

void stream_base::set_memcpy(memcpy_function memcpy)
{
    set_memcpy(SPEAD2_ADAPT_MEMCPY(memcpy, memcpy));
}

void stream_base::set_memcpy(memcpy_function_id id)
{
    /* We adapt each case to the packet_memcpy signature rather than using the
     * generic wrapping in the memcpy_function overload. This ensures that
     * there is only one level of indirect function call instead of two.
     */
    switch (id)
    {
    case MEMCPY_STD:
        set_memcpy(SPEAD2_ADAPT_MEMCPY(std::memcpy, ));
        break;
    case MEMCPY_NONTEMPORAL:
        set_memcpy(SPEAD2_ADAPT_MEMCPY(spead2::memcpy_nontemporal, ));
        break;
    default:
        throw std::invalid_argument("Unknown memcpy function");
    }
}

void stream_base::set_stop_on_stop_item(bool stop)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    stop_on_stop_item = stop;
}

bool stream_base::get_stop_on_stop_item() const
{
    std::lock_guard<std::mutex> lock(config_mutex);
    return stop_on_stop_item;
}

void stream_base::set_allow_unsized_heaps(bool allow)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    allow_unsized_heaps = allow;
}

bool stream_base::get_allow_unsized_heaps() const
{
    std::lock_guard<std::mutex> lock(config_mutex);
    return allow_unsized_heaps;
}

stream_base::add_packet_state::add_packet_state(stream_base &owner)
    : owner(owner), lock(owner.queue_mutex)
{
    std::lock_guard<std::mutex> config_lock(owner.config_mutex);
    allocator = owner.allocator;
    memcpy = owner.memcpy;
    stop_on_stop_item = owner.stop_on_stop_item;
    allow_unsized_heaps = owner.allow_unsized_heaps;
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
    assert(!stopped);
    state.packets++;
    if (packet.heap_length < 0 && !state.allow_unsized_heaps)
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
        // Never seen this heap before. Evict the old one in its slot,
        // if any. Note: not safe to dereference h just anywhere here!
        if (++head == max_heaps)
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
        new (&entry->heap) live_heap(packet, bug_compat);
    }

    live_heap *h = &entry->heap;
    bool result = false;
    bool end_of_stream = false;
    if (h->add_packet(packet, state.memcpy, *state.allocator))
    {
        result = true;
        end_of_stream = state.stop_on_stop_item && h->is_end_of_stream();
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


stream::stream(io_service_ref io_service, bug_compat_mask bug_compat, std::size_t max_heaps)
    : stream_base(bug_compat, max_heaps),
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
