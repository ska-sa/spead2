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
 * This is an example of using the chunking receive API with ringbuffers. To
 * test it, run
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
#include <spead2/common_ringbuffer.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_chunk_stream.h>
#include <spead2/recv_udp.h>

static constexpr std::size_t heap_payload_size = 65536;
static constexpr std::size_t heaps_per_chunk = 64;
static constexpr std::size_t chunk_payload_size = heaps_per_chunk * heap_payload_size;

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

int main()
{
    constexpr int max_chunks = 4;
    auto chunk_config = spead2::recv::chunk_stream_config()
        .set_items({spead2::HEAP_CNT_ID, spead2::HEAP_LENGTH_ID})
        .set_max_chunks(max_chunks)
        .set_place(chunk_place);
    auto stream_config = spead2::recv::stream_config();
    using chunk_ringbuffer = spead2::ringbuffer<std::unique_ptr<spead2::recv::chunk>>;
    auto data_ring = std::make_shared<chunk_ringbuffer>(max_chunks);
    auto free_ring = std::make_shared<chunk_ringbuffer>(max_chunks);
    auto allocator = std::make_shared<spead2::memory_allocator>();

    spead2::thread_pool worker;
    spead2::recv::chunk_ring_stream<> stream(worker, stream_config, chunk_config, data_ring, free_ring);
    for (int i = 0; i < max_chunks; i++)
    {
        auto chunk = std::make_unique<spead2::recv::chunk>();
        chunk->present = allocator->allocate(heaps_per_chunk, nullptr);
        chunk->present_size = heaps_per_chunk;
        chunk->data = allocator->allocate(chunk_payload_size, nullptr);
        stream.add_free_chunk(std::move(chunk));
    }

    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::any(), 8888);
    stream.emplace_reader<spead2::recv::udp_reader>(
        endpoint, spead2::recv::udp_reader::default_max_size, 1024 * 1024);
    while (true)
    {
        try
        {
            auto chunk = data_ring->pop();
            auto n_present = std::accumulate(
                chunk->present.get(),
                chunk->present.get() + chunk->present_size, std::size_t(0));
            std::cout << "Received chunk " << chunk->chunk_id << " with "
                << n_present << " / " << heaps_per_chunk << " heaps\n";
            stream.add_free_chunk(std::move(chunk));
        }
        catch (spead2::ringbuffer_stopped &)
        {
            break;
        }
    }

    return 0;
}
