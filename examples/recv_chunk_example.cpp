/* Copyright 2021, 2023 National Research Foundation (SARAO)
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
 *
 * This is an example of using the chunking receive API. To test it, run
 * <code>spead2_send localhost:8888 --heaps 1000 --heap-size 65536 --rate 10</code>.
 */

#include <iostream>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <algorithm>
#include <numeric>
#include <future>
#include <spead2/common_thread_pool.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_chunk_stream.h>
#include <spead2/recv_udp.h>

static constexpr std::size_t heap_payload_size = 65536;
static constexpr std::size_t heaps_per_chunk = 64;
static constexpr std::size_t chunk_payload_size = heaps_per_chunk * heap_payload_size;

static std::shared_ptr<spead2::memory_allocator> allocator;

class joinable_chunk_stream : public spead2::recv::chunk_stream
{
private:
    std::promise<void> stop_promise;

public:
    using spead2::recv::chunk_stream::chunk_stream;

    virtual void stop_received() override
    {
        spead2::recv::chunk_stream::stop_received();
        stop_promise.set_value();
    }

    void join()
    {
        std::future<void> future = stop_promise.get_future();
        future.get();
    }
};

static void chunk_place(spead2::recv::chunk_place_data *data, [[maybe_unused]] std::size_t data_size)
{
    // We requested only the heap ID and size
    auto heap_cnt = data->items[0];
    auto payload_size = data->items[1];
    // If the payload size doesn't match, discard the heap (could be descriptors etc).
    if (payload_size == heap_payload_size)
    {
        data->chunk_id = heap_cnt / heaps_per_chunk;
        data->heap_index = heap_cnt % heaps_per_chunk;
        data->heap_offset = data->heap_index * heap_payload_size;
    }
}

static std::unique_ptr<spead2::recv::chunk> chunk_allocate(
    [[maybe_unused]] std::int64_t chunk_id, [[maybe_unused]] std::uint64_t *batch_stats)
{
    auto chunk = std::make_unique<spead2::recv::chunk>();
    chunk->present = allocator->allocate(heaps_per_chunk, nullptr);
    chunk->present_size = heaps_per_chunk;
    // Indicate that no heaps are present yet
    std::fill(chunk->present.get(), chunk->present.get() + heaps_per_chunk, 0);
    chunk->data = allocator->allocate(chunk_payload_size, nullptr);
    return chunk;
}

static void chunk_ready(
    std::unique_ptr<spead2::recv::chunk> &&chunk, [[maybe_unused]] std::uint64_t *batch_stats)
{
    auto n_present = std::accumulate(
        chunk->present.get(),
        chunk->present.get() + heaps_per_chunk, std::size_t(0));
    std::cout << "Received chunk " << chunk->chunk_id << " with "
        << n_present << " / " << heaps_per_chunk << " heaps\n";
}

int main()
{
    auto chunk_config = spead2::recv::chunk_stream_config()
        .set_items({spead2::HEAP_CNT_ID, spead2::HEAP_LENGTH_ID})
        .set_max_chunks(4)
        .set_place(chunk_place)
        .set_allocate(chunk_allocate)
        .set_ready(chunk_ready);
    auto stream_config = spead2::recv::stream_config();

    allocator = std::make_shared<spead2::memory_allocator>();
    spead2::thread_pool worker;
    joinable_chunk_stream stream(worker, stream_config, chunk_config);
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::any(), 8888);
    stream.emplace_reader<spead2::recv::udp_reader>(
        endpoint, spead2::recv::udp_reader::default_max_size, 1024 * 1024);
    stream.join();

    return 0;
}
