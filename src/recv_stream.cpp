/* Copyright 2015, 2017-2021, 2023, 2025 National Research Foundation (SARAO)
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
#include <new>
#include <spead2/recv_stream.h>
#include <spead2/recv_live_heap.h>
#include <spead2/common_memcpy.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_logging.h>

#define INVALID_ENTRY ((queue_entry *) -1)

namespace spead2::recv
{

stream_stat_config::stream_stat_config(std::string name, mode mode_)
    : name(std::move(name)), mode_(mode_)
{
}

std::uint64_t stream_stat_config::combine(std::uint64_t a, std::uint64_t b) const
{
    switch (mode_)
    {
    case mode::COUNTER:
        return a + b;
    case mode::MAXIMUM:
        return std::max(a, b);
    }
    /* Line below should normally be unreachable. Using the same expression as
     * for COUNTER lets the compiler generate more efficient code, as it only
     * has to consider two cases (looks just as good as using GCC's
     * __builtin_unreachable, without depending on compiler specifics).
     */
    return a + b;   // LCOV_EXCL_LINE
}

bool operator==(const stream_stat_config &a, const stream_stat_config &b)
{
    return a.get_name() == b.get_name() && a.get_mode() == b.get_mode();
}

bool operator!=(const stream_stat_config &a, const stream_stat_config &b)
{
    return !(a == b);
}

/**
 * Get the index within @a stats corresponding to @a name. If it is not
 * present, returns @c stats.size().
 */
static std::size_t get_stat_index_nothrow(
    const std::vector<stream_stat_config> &stats,
    const std::string &name)
{
    for (std::size_t i = 0; i < stats.size(); i++)
        if (stats[i].get_name() == name)
            return i;
    return stats.size();
}

/**
 * Get the index within @a stats corresponding to @a name.
 *
 * @throw std::out_of_range if it is not present
 */
static std::size_t get_stat_index(
    const std::vector<stream_stat_config> &stats,
    const std::string &name)
{
    std::size_t ret = get_stat_index_nothrow(stats, name);
    if (ret == stats.size())
        throw std::out_of_range(name + " is not a known statistic name");
    return ret;
}


static std::shared_ptr<const std::vector<stream_stat_config>> make_default_stats()
{
    auto stats = std::make_shared<std::vector<stream_stat_config>>();
    // Keep this in sync with the stream_stat_* constexprs in the header
    stats->emplace_back("heaps", stream_stat_config::mode::COUNTER);
    stats->emplace_back("incomplete_heaps_evicted", stream_stat_config::mode::COUNTER);
    stats->emplace_back("incomplete_heaps_flushed", stream_stat_config::mode::COUNTER);
    stats->emplace_back("packets", stream_stat_config::mode::COUNTER);
    stats->emplace_back("batches", stream_stat_config::mode::COUNTER);
    stats->emplace_back("max_batch", stream_stat_config::mode::MAXIMUM);
    stats->emplace_back("single_packet_heaps", stream_stat_config::mode::COUNTER);
    stats->emplace_back("search_dist", stream_stat_config::mode::COUNTER);
    // For backwards compatibility, worker_blocked is always stats->emplace_backed, although
    // it is not part of the base stream statistics
    stats->emplace_back("worker_blocked", stream_stat_config::mode::COUNTER);
    assert(stats->size() == stream_stat_indices::custom);
    return stats;
}

/* This is used for stream_stats objects that do not have any custom statistics.
 * Sharing this means the compatibility check for operator+ requires only a
 * pointer comparison rather than comparing arrays.
 */
static std::shared_ptr<const std::vector<stream_stat_config>> default_stats = make_default_stats();

stream_stats::stream_stats()
    : stream_stats(default_stats)
{
}

stream_stats::stream_stats(std::shared_ptr<const std::vector<stream_stat_config>> config)
    : stream_stats(config, std::vector<std::uint64_t>(config->size()))
{
    // Note: annoyingly, can't use std::move(config) above, because we access
    // config to get the size to use for the vector.
}

stream_stats::stream_stats(std::shared_ptr<const std::vector<stream_stat_config>> config,
                           std::vector<std::uint64_t> values)
    : config(std::move(config)),
    values(std::move(values)),
    heaps(this->values[stream_stat_indices::heaps]),
    incomplete_heaps_evicted(this->values[stream_stat_indices::incomplete_heaps_evicted]),
    incomplete_heaps_flushed(this->values[stream_stat_indices::incomplete_heaps_flushed]),
    packets(this->values[stream_stat_indices::packets]),
    batches(this->values[stream_stat_indices::batches]),
    worker_blocked(this->values[stream_stat_indices::worker_blocked]),
    max_batch(this->values[stream_stat_indices::max_batch]),
    single_packet_heaps(this->values[stream_stat_indices::single_packet_heaps]),
    search_dist(this->values[stream_stat_indices::search_dist])
{
    assert(this->config->size() >= stream_stat_indices::custom);
    assert(this->config->size() == this->values.size());
}

stream_stats::stream_stats(const stream_stats &other)
    : stream_stats(other.config, other.values)
{
}

stream_stats &stream_stats::operator=(const stream_stats &other)
{
    if (config != other.config && *config != *other.config)
        throw std::invalid_argument("config must match to assign stats");
    for (std::size_t i = 0; i < values.size(); i++)
        values[i] = other.values[i];
    return *this;
}

std::uint64_t &stream_stats::operator[](const std::string &name)
{
    return at(name);
}

const std::uint64_t &stream_stats::operator[](const std::string &name) const
{
    return at(name);
}

std::uint64_t &stream_stats::at(const std::string &name)
{
    return values[get_stat_index(*config, name)];
}

const std::uint64_t &stream_stats::at(const std::string &name) const
{
    return values[get_stat_index(*config, name)];
}

stream_stats::iterator stream_stats::find(const std::string &name)
{
    return iterator(*this, get_stat_index_nothrow(*config, name));
}

stream_stats::const_iterator stream_stats::find(const std::string &name) const
{
    return const_iterator(*this, get_stat_index_nothrow(*config, name));
}

std::size_t stream_stats::count(const std::string &name) const
{
    return get_stat_index_nothrow(*config, name) != values.size() ? 1 : 0;
}

stream_stats stream_stats::operator+(const stream_stats &other) const
{
    stream_stats out = *this;
    out += other;
    return out;
}

stream_stats &stream_stats::operator+=(const stream_stats &other)
{
    if (config != other.config && *config != *other.config)
        throw std::invalid_argument("config must match to add stats together");
    for (std::size_t i = 0; i < values.size(); i++)
        values[i] = (*config)[i].combine(values[i], other.values[i]);
    return *this;
}


static std::size_t compute_bucket_count(std::size_t total_max_heaps)
{
    std::size_t buckets = 4;
    while (buckets < total_max_heaps)
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
    allocator(std::make_shared<memory_allocator>()),
    stats(default_stats)  // Initially point to shared defaults; make a copy on write
{
}

stream_config &stream_config::set_max_heaps(std::size_t max_heaps)
{
    if (max_heaps == 0)
        throw std::invalid_argument("max_heaps cannot be 0");
    this->max_heaps = max_heaps;
    return *this;
}

stream_config &stream_config::set_substreams(std::size_t substreams)
{
    if (substreams == 0)
        throw std::invalid_argument("substreams cannot be 0");
    this->substreams = substreams;
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

stream_config &stream_config::set_stream_id(std::uintptr_t id)
{
    stream_id = id;
    return *this;
}

stream_config &stream_config::set_explicit_start(bool explicit_start)
{
    this->explicit_start = explicit_start;
    return *this;
}

std::size_t stream_config::add_stat(std::string name, stream_stat_config::mode mode)
{
    if (spead2::recv::get_stat_index_nothrow(*stats, name) != stats->size())
        throw std::invalid_argument("A statistic called " + name + " already exists");
    // Make a copy so that we don't modify any shared copies
    auto new_stats = std::make_shared<std::vector<stream_stat_config>>(*stats);
    std::size_t index = new_stats->size();
    new_stats->emplace_back(std::move(name), mode);
    stats = std::move(new_stats);
    return index;
}

std::size_t stream_config::get_stat_index(const std::string &name) const
{
    return spead2::recv::get_stat_index(*stats, name);
}


stream_base::stream_base(const stream_config &config)
    : queue_storage(new queue_entry[config.get_max_heaps() * config.get_substreams()]),
    bucket_count(compute_bucket_count(config.get_max_heaps() * config.get_substreams())),
    bucket_shift(compute_bucket_shift(bucket_count)),
    buckets(new queue_entry *[bucket_count]),
    substreams(new substream[config.get_substreams() + 1]),
    substream_div(config.get_substreams()),
    config(config),
    shared(std::make_shared<shared_state>(this)),
    stats(config.get_stats().size()),
    batch_stats(config.get_stats().size())
{
    for (std::size_t i = 0; i < config.get_max_heaps() * config.get_substreams(); i++)
        queue_storage[i].next = INVALID_ENTRY;
    for (std::size_t i = 0; i < bucket_count; i++)
        buckets[i] = NULL;
    for (std::size_t i = 0; i <= config.get_substreams(); i++)
    {
        substreams[i].start = i * config.get_max_heaps();
        substreams[i].head = substreams[i].start;
    }
}

stream_base::~stream_base()
{
    for (std::size_t i = 0; i < get_config().get_max_heaps() * get_config().get_substreams(); i++)
    {
        queue_entry *entry = &queue_storage[i];
        if (entry->next != INVALID_ENTRY)
        {
            unlink_entry(entry);
            entry->heap.destroy();
        }
    }
}

std::size_t stream_base::get_bucket(item_pointer_t heap_cnt) const
{
    // Look up Fibonacci hashing for an explanation of the magic number
    return (heap_cnt * 11400714819323198485ULL) >> bucket_shift;
}

std::size_t stream_base::get_substream(item_pointer_t heap_cnt) const
{
    // libdivide doesn't provide operator %
    return heap_cnt - (heap_cnt / substream_div * config.get_substreams());
}

void stream_base::unlink_entry(queue_entry *entry)
{
    assert(entry->next != INVALID_ENTRY);
    std::size_t bucket_id = get_bucket(entry->heap->get_cnt());
    queue_entry **prev = &buckets[bucket_id];
    while (*prev != entry)
    {
        assert(*prev != NULL && *prev != INVALID_ENTRY);
        prev = &(*prev)->next;
    }
    *prev = entry->next;
    entry->next = INVALID_ENTRY;
}

stream_base::add_packet_state::add_packet_state(shared_state &owner)
    : lock(owner.queue_mutex), owner(owner.self)
{
    if (this->owner)
    {
        stopped = this->owner->stopped;
        std::fill(this->owner->batch_stats.begin(), this->owner->batch_stats.end(), 0);
    }
    else
    {
        stopped = true;
    }
}

stream_base::add_packet_state::~add_packet_state()
{
    if (owner && stopped)
        owner->stop_received();
    if (!owner || (!packets && is_stopped()))
        return;   // Stream was stopped before we could do anything - don't count as a batch
    std::lock_guard<std::mutex> stats_lock(owner->stats_mutex);
    // The built-in stats are updated directly; batch_stats is not used
    owner->stats[stream_stat_indices::packets] += packets;
    owner->stats[stream_stat_indices::batches]++;
    owner->stats[stream_stat_indices::heaps] += complete_heaps + incomplete_heaps_evicted;
    owner->stats[stream_stat_indices::incomplete_heaps_evicted] += incomplete_heaps_evicted;
    owner->stats[stream_stat_indices::single_packet_heaps] += single_packet_heaps;
    owner->stats[stream_stat_indices::search_dist] += search_dist;
    auto &owner_max_batch = owner->stats[stream_stat_indices::max_batch];
    owner_max_batch = std::max(owner_max_batch, packets);
    // Update custom statistics
    const auto &stats_config = owner->get_config().get_stats();
    for (std::size_t i = stream_stat_indices::custom; i < stats_config.size(); i++)
        owner->stats[i] = stats_config[i].combine(owner->stats[i], owner->batch_stats[i]);
}

bool stream_base::add_packet(add_packet_state &state, const packet_header &packet)
{
    const stream_config &config = state.owner->get_config();
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
            if (entry->heap->get_cnt() == heap_cnt)
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

        std::size_t substream_id = get_substream(heap_cnt);
        substream &ss = substreams[substream_id];
        if (++ss.head == substreams[substream_id + 1].start)
            ss.head = ss.start;
        entry = &queue_storage[ss.head];
        if (entry->next != INVALID_ENTRY)
        {
            state.incomplete_heaps_evicted++;
            unlink_entry(entry);
            heap_ready(std::move(*entry->heap));
            entry->heap.destroy();
        }
        entry->next = buckets[bucket_id];
        buckets[bucket_id] = entry;
        entry->heap.construct(packet, config.get_bug_compat());
    }

    live_heap *h = entry->heap.get();
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
            entry->heap.destroy();
        }
    }

    if (end_of_stream)
        state.stop();
    return result;
}

void stream_base::flush_unlocked()
{
    const std::size_t num_substreams = get_config().get_substreams();
    std::size_t n_flushed = 0;
    for (std::size_t i = 0; i < num_substreams; i++)
    {
        substream &ss = substreams[i];
        const std::size_t end = substreams[i + 1].start;
        for (std::size_t j = ss.start; j < end; j++)
        {
            if (++ss.head == substreams[i + 1].start)
                ss.head = ss.start;
            queue_entry *entry = &queue_storage[ss.head];
            if (entry->next != INVALID_ENTRY)
            {
                n_flushed++;
                unlink_entry(entry);
                heap_ready(std::move(*entry->heap));
                entry->heap.destroy();
            }
        }
    }
    std::lock_guard<std::mutex> stats_lock(stats_mutex);
    stats[stream_stat_indices::heaps] += n_flushed;
    stats[stream_stat_indices::incomplete_heaps_flushed] += n_flushed;
}

void stream_base::flush()
{
    std::lock_guard<std::mutex> lock(shared->queue_mutex);
    flush_unlocked();
}

void stream_base::stop_unlocked()
{
    if (!stopped)
        stop_received();
}

void stream_base::stop()
{
    std::lock_guard<std::mutex> lock(shared->queue_mutex);
    stop_unlocked();
}

void stream_base::stop_received()
{
    assert(!stopped);
    stopped = true;
    shared->self = nullptr;
    flush_unlocked();
}

stream_stats stream_base::get_stats() const
{
    std::lock_guard<std::mutex> stats_lock(stats_mutex);
    stream_stats ret(get_config().stats, stats);
    return ret;
}


reader::reader(stream &owner)
    : io_context(owner.get_io_context()), owner(owner.shared)
{
}

bool reader::lossy() const
{
    return true;
}


stream::stream(io_context_ref io_context, const stream_config &config)
    : stream_base(config),
    thread_pool_holder(std::move(io_context).get_shared_thread_pool()),
    io_context(*io_context),
    readers_started(!config.get_explicit_start())
{
}

void stream::start()
{
    std::lock_guard<std::mutex> lock(reader_mutex);
    if (!readers_started)
    {
        for (const auto &r : readers)
            r->start();
        readers_started = true;
    }
}

void stream::stop_received()
{
    stream_base::stop_received();
    std::lock_guard<std::mutex> lock(reader_mutex);
    for (const auto &r : readers)
        r->stop();
    /* This ensures that once we stop the readers, any future call to
     * emplace_reader will silently be ignored. This avoids issues if there
     * is a race between the user calling emplace_reader and a stop packet
     * in the stream.
     */
    stop_readers = true;
}

void stream::stop_impl()
{
    stream_base::stop();
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


const std::uint8_t *mem_to_stream(stream_base::add_packet_state &state, const std::uint8_t *ptr, std::size_t length)
{
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

const std::uint8_t *mem_to_stream(stream_base &s, const std::uint8_t *ptr, std::size_t length)
{
    stream_base::add_packet_state state(s);
    return mem_to_stream(state, ptr, length);
}

} // namespace spead2::recv
