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
    group.release_chunk(c);
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

void chunk_stream_group::stop()
{
    /* Streams will try to lock the group (and modify `streams`) while
     * stopping, so we move the streams set into a local variable.
     *
     * The mutex is not held while stopping streams, so streams can
     * asynchronously stop under us. That's okay because the contract for this
     * function is that it's not allowed to occur concurrently with destroying
     * streams.
     */
    std::unique_lock<std::mutex> lock(mutex);
    auto streams_local = std::move(streams);
    lock.unlock();
    for (auto stream : streams_local)
        stream->stop();

    lock.lock();
    while (chunks.get_head_chunk() != chunks.get_tail_chunk())
        chunks.flush_head([this](chunk *c) { ready_chunk(c, nullptr); });
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
    if (chunk_id >= chunks.get_head_chunk() + max_chunks)
    {
        std::int64_t target = chunk_id - max_chunks + 1;  // first chunk we don't need to flush
        for (chunk_stream_group_member *s : streams)
            s->async_flush_until(target);
        std::int64_t to_check = chunks.get_head_chunk(); // next chunk to wait for
        while (true)
        {
            bool good = true;
            std::int64_t limit = std::min(chunks.get_tail_chunk(), target);
            to_check = std::max(chunks.get_head_chunk(), to_check);
            for (; to_check < limit; to_check++)
            {
                chunk *c = chunks.get_chunk(to_check);
                if (c && c->ref_count > 0)
                {
                    good = false;  // Still need to wait longer for this chunk
                    break;
                }
            }
            if (good)
                break;
            ready_condition.wait(lock);
        }
    }

    chunk *c = chunks.get_chunk(
        chunk_id,
        stream_id,
        [this, batch_stats](std::int64_t id) {
            return config.get_allocate()(id, batch_stats).release();
        },
        [this, batch_stats](chunk *c) { ready_chunk(c, batch_stats); }
    );
    if (c)
        c->ref_count++;
    return c;
}

void chunk_stream_group::ready_chunk(chunk *c, std::uint64_t *batch_stats)
{
    std::unique_ptr<chunk> owned(c);
    config.get_ready()(std::move(owned), batch_stats);
}

void chunk_stream_group::release_chunk(chunk *c)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (--c->ref_count == 0)
        ready_condition.notify_all();
}

void chunk_stream_group::stream_added(chunk_stream_group_member &s)
{
    std::lock_guard<std::mutex> lock(mutex);
    bool added = streams.insert(&s).second;
    assert(added); // should be impossible to add the same stream twice
    (void) added; // suppress warning when NDEBUG is defined
}

void chunk_stream_group::stream_stop_received(chunk_stream_group_member &s)
{
    std::lock_guard<std::mutex> lock(mutex);
    streams.erase(&s);
}


chunk_stream_group_member::chunk_stream_group_member(
    io_service_ref io_service,
    const stream_config &config,
    const chunk_stream_config &chunk_config,
    chunk_stream_group &group)
    : chunk_stream_state(config, chunk_config, detail::chunk_manager_group(group)),
    stream(std::move(io_service), adjust_config(config)),
    group(group)
{
    group.stream_added(*this);
}

void chunk_stream_group_member::heap_ready(live_heap &&lh)
{
    do_heap_ready(std::move(lh));
}

void chunk_stream_group_member::async_flush_until(std::int64_t chunk_id)
{
    std::shared_ptr<stream_base::shared_state> shared = this->shared;
    // TODO: once we depend on C++14, move rather than copying into the lambda
    boost::asio::post(get_io_service(), [shared, chunk_id]() {
        std::lock_guard<std::mutex> lock(shared->queue_mutex);
        if (!shared->self)
            return; // We've stopped, which means everything is flushed
        chunk_stream_group_member *self = static_cast<chunk_stream_group_member *>(shared->self);
        while (self->chunks.get_head_chunk() < chunk_id)
        {
            self->chunks.flush_head([self](chunk *c) {
                self->group.release_chunk(c);
            });
        }
    });
}

void chunk_stream_group_member::stop_received()
{
    stream::stop_received();
    group.stream_stop_received(*this);
    flush_chunks();
}

void chunk_stream_group_member::stop()
{
    group.stream_pre_stop(*this);
    {
        std::lock_guard<std::mutex> lock(shared->queue_mutex);
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
