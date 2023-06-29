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

#include <cstddef>
#include <vector>
#include <mutex>
#include <spead2/recv_chunk_stream.h>
#include <spead2/recv_chunk_stream_group.h>

namespace spead2
{
namespace recv
{

constexpr std::size_t chunk_stream_group_config::default_max_chunks;

chunk_stream_group_config &chunk_stream_group_config::set_max_chunks(std::size_t max_chunks)
{
    if (max_chunks == 0)
        throw std::invalid_argument("max_chunks cannot be 0");
    this->max_chunks = max_chunks;
    return *this;
}

chunk_stream_group_config &chunk_stream_group_config::set_eviction_mode(eviction_mode eviction_mode_)
{
    this->eviction_mode_ = eviction_mode_;
    return *this;
}

chunk_stream_group_config &chunk_stream_group_config::set_allocate(chunk_allocate_function allocate)
{
    this->allocate = std::move(allocate);
    return *this;
}

chunk_stream_group_config &chunk_stream_group_config::set_ready(chunk_ready_function ready)
{
    this->ready = std::move(ready);
    return *this;
}

namespace detail
{

chunk_manager_group::chunk_manager_group(chunk_stream_group &group)
    : group(group)
{
}

std::uint64_t *chunk_manager_group::get_batch_stats(chunk_stream_state<chunk_manager_group> &state) const
{
    return static_cast<chunk_stream_group_member *>(&state)->batch_stats.data();
}

chunk *chunk_manager_group::allocate_chunk(
    chunk_stream_state<chunk_manager_group> &state, std::int64_t chunk_id)
{
    return group.get_chunk(chunk_id, state.stream_id, state.place_data->batch_stats);
}

void chunk_manager_group::ready_chunk(chunk_stream_state<chunk_manager_group> &state, chunk *c)
{
    std::uint64_t *batch_stats = static_cast<chunk_stream_group_member *>(&state)->batch_stats.data();
    group.release_chunk(c, batch_stats);
}

} // namespace detail

chunk_stream_group::chunk_stream_group(const chunk_stream_group_config &config)
    : config(config), chunks(config.get_max_chunks())
{
}

chunk_stream_group::~chunk_stream_group()
{
    stop();
}

chunk_stream_group::iterator chunk_stream_group::begin() noexcept
{
    return iterator(streams.begin());
}

chunk_stream_group::iterator chunk_stream_group::end() noexcept
{
    return iterator(streams.end());
}

chunk_stream_group::const_iterator chunk_stream_group::begin() const noexcept
{
    return const_iterator(streams.begin());
}

chunk_stream_group::const_iterator chunk_stream_group::end() const noexcept
{
    return const_iterator(streams.end());
}

chunk_stream_group::const_iterator chunk_stream_group::cbegin() const noexcept
{
    return const_iterator(streams.begin());
}

chunk_stream_group::const_iterator chunk_stream_group::cend() const noexcept
{
    return const_iterator(streams.end());
}

chunk_stream_group_member &chunk_stream_group::emplace_back(
    io_service_ref io_service,
    const stream_config &config,
    const chunk_stream_config &chunk_config)
{
    return emplace_back<chunk_stream_group_member>(std::move(io_service), config, chunk_config);
}

void chunk_stream_group::stop()
{
    /* The mutex is not held while stopping streams, so that callbacks
     * triggered by stopping the streams can take the lock if necessary.
     *
     * It's safe to iterate streams without the mutex because this function
     * is called by the user, so a simultaneous call to emplace_back would
     * violate the requirement that the user doesn't call the API from more
     * than one thread at a time.
     *
     * The last stream to stop will flush the window (see
     * stream_stop_received).
     */
    for (const auto &stream : streams)
        stream->stop();
}

void chunk_stream_group::stream_stop_received(chunk_stream_group_member &s)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (--live_streams == 0)
    {
        // Once all the streams have stopped, make all the chunks in the
        // window available. It's not necessary to check c->ref_count
        // because no stream can have a reference once they're all stopped.
        std::uint64_t *batch_stats = s.batch_stats.data();
        while (!chunks.empty())
        {
            chunks.flush_head([this, batch_stats](chunk *c) { ready_chunk(c, batch_stats); });
        }
    }
}

chunk *chunk_stream_group::get_chunk(std::int64_t chunk_id, std::uintptr_t stream_id, std::uint64_t *batch_stats)
{
    std::unique_lock<std::mutex> lock(mutex);
    /* Any chunk old enough be made ready needs to first be released by the
     * member streams. To do that, we request all the streams to flush, then
     * wait until it is safe, using the condition variable to wake up
     * whenever there is forward progress.
     *
     * Another thread may call get_chunk in the meantime and advance the
     * window, so we must be careful not to assume anything about the
     * state after a wait.
     */
    const std::size_t max_chunks = config.get_max_chunks();
    if (chunk_id >= chunks.get_head_chunk() + std::int64_t(max_chunks))
    {
        std::int64_t target = chunk_id - max_chunks + 1;  // first chunk we don't need to flush
        if (config.get_eviction_mode() == chunk_stream_group_config::eviction_mode::LOSSY)
            for (const auto &s : streams)
                s->async_flush_until(target);
        while (chunks.get_head_chunk() < std::min(chunks.get_tail_chunk(), target))
        {
            chunk *c = chunks.get_chunk(chunks.get_head_chunk());
            if (c->ref_count == 0)
                chunks.flush_head([this, batch_stats](chunk *c2) { ready_chunk(c2, batch_stats); });
            else
                ready_condition.wait(lock);
        }
    }

    chunk *c = chunks.get_chunk(
        chunk_id,
        stream_id,
        [this, batch_stats](std::int64_t id) {
            return config.get_allocate()(id, batch_stats).release();
        },
        [](chunk *) {
            // Should be unreachable, as we've done the necessary flushing above
            assert(false);
        }
    );
    if (c)
        c->ref_count++;
    return c;
}

void chunk_stream_group::ready_chunk(chunk *c, std::uint64_t *batch_stats)
{
    assert(c->ref_count == 0);
    std::unique_ptr<chunk> owned(c);
    config.get_ready()(std::move(owned), batch_stats);
}

void chunk_stream_group::release_chunk(chunk *c, std::uint64_t *batch_stats)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (--c->ref_count == 0)
        ready_condition.notify_all();
}

chunk_stream_group_member::chunk_stream_group_member(
    chunk_stream_group &group,
    io_service_ref io_service,
    const stream_config &config,
    const chunk_stream_config &chunk_config)
    : chunk_stream_state(config, chunk_config, detail::chunk_manager_group(group)),
    stream(std::move(io_service), adjust_config(config)),
    group(group)
{
    if (chunk_config.get_max_chunks() > group.config.get_max_chunks())
        throw std::invalid_argument("stream max_chunks must not be larger than group max_chunks");
}

void chunk_stream_group_member::heap_ready(live_heap &&lh)
{
    do_heap_ready(std::move(lh));
}

void chunk_stream_group_member::async_flush_until(std::int64_t chunk_id)
{
    post([chunk_id](stream_base &s) {
        chunk_stream_group_member &self = static_cast<chunk_stream_group_member &>(s);
        self.chunks.flush_until(chunk_id, [&self](chunk *c) {
            self.group.release_chunk(c, self.batch_stats.data());
        });
    });
}

void chunk_stream_group_member::stop_received()
{
    stream::stop_received();
    flush_chunks();
    group.stream_stop_received(*this);
}

void chunk_stream_group_member::stop()
{
    group.stream_pre_stop(*this);
    {
        std::lock_guard<std::mutex> lock(get_queue_mutex());
        flush_chunks();
    }
    stream::stop();
}

chunk_stream_group_member::~chunk_stream_group_member()
{
    stop();
}

} // namespace recv
} // namespace spead2
