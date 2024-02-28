/* Copyright 2023-2024 National Research Foundation (SARAO)
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

#include <cstdint>
#include <cstddef>
#include <utility>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <numeric>
#include <unistd.h>
#include <boost/asio.hpp>
#include <spead2/common_defines.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_thread_pool.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_chunk_stream.h>
#include <spead2/recv_udp.h>

#if defined(__GNUC__) && defined(__x86_64__)
// Compile this function with AVX2 for better performance. Remove this if your
// CPU does not support AVX2 (e.g., if you get an Illegal Instruction error).
[[gnu::target("avx2")]]
#endif
static double mean_power(const std::int8_t *adc_samples, const std::uint8_t *present,
                         std::size_t heap_size, std::size_t heaps)
{
    std::int64_t sum = 0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < heaps; i++)
    {
        if (present[i])
        {
            for (std::size_t j = 0; j < heap_size; j++)
            {
                std::int64_t sample = adc_samples[i * heap_size + j];
                sum += sample * sample;
            }
            n += heap_size;
        }
    }
    return double(sum) / n;
}

void place_callback(
    spead2::recv::chunk_place_data *data,
    std::int64_t heap_size, std::int64_t chunk_size)
{
    auto payload_size = data->items[0];
    auto timestamp = data->items[1];
    if (timestamp >= 0 && payload_size == heap_size)
    {
        data->chunk_id = timestamp / chunk_size;
        data->heap_offset = timestamp % chunk_size;
        data->heap_index = data->heap_offset / heap_size;
    }
}

static void usage(const char *name)
{
    std::cerr << "Usage: " << name << " [-H heap-size] port\n";
}

int main(int argc, char * const argv[])
{
    int opt;
    std::int64_t heap_size = 1024 * 1024;
    while ((opt = getopt(argc, argv, "H:")) != -1)
    {
        switch (opt)
        {
        case 'H':
            heap_size = std::stoll(optarg);
            break;
        default:
            usage(argv[0]);
            return 2;
        }
    }
    if (argc - optind != 1)
    {
        usage(argv[0]);
        return 2;
    }

    std::int64_t chunk_size = 1024 * 1024;  // Preliminary value
    std::int64_t chunk_heaps = std::max(std::int64_t(1), chunk_size / heap_size);
    chunk_size = chunk_heaps * heap_size;  // Final value

    spead2::thread_pool thread_pool(1, {2});
    spead2::thread_pool::set_affinity(3);
    spead2::recv::stream_config config;
    config.set_max_heaps(2);
    spead2::recv::chunk_stream_config chunk_config;
    chunk_config.set_items({spead2::HEAP_LENGTH_ID, 0x1600});
    chunk_config.set_max_chunks(1);
    chunk_config.set_place(
        [=](auto data, auto) { place_callback(data, heap_size, chunk_size); }
    );

    using ringbuffer = spead2::ringbuffer<std::unique_ptr<spead2::recv::chunk>>;
    auto data_ring = std::make_shared<ringbuffer>(2);
    auto free_ring = std::make_shared<ringbuffer>(4);
    spead2::recv::chunk_ring_stream stream(
        thread_pool, config, chunk_config, data_ring, free_ring
    );
    for (std::size_t i = 0; i < free_ring->capacity(); i++)
    {
        auto chunk = std::make_unique<spead2::recv::chunk>();
        chunk->present = std::make_unique<std::uint8_t[]>(chunk_heaps);
        chunk->present_size = chunk_heaps;
        chunk->data = std::make_unique<std::uint8_t[]>(chunk_size);
        stream.add_free_chunk(std::move(chunk));
    }

    boost::asio::ip::udp::endpoint endpoint(
        boost::asio::ip::address_v4::any(), std::stoi(argv[optind]));
    stream.emplace_reader<spead2::recv::udp_reader>(endpoint);
    std::int64_t n_heaps = 0;
    for (std::unique_ptr<spead2::recv::chunk> chunk : *data_ring)
    {
        auto present = chunk->present.get();
        auto n = std::accumulate(present, present + chunk_heaps, std::size_t(0));
        if (n > 0)
        {
            std::int64_t timestamp = chunk->chunk_id * chunk_size;
            auto adc_samples = (const std::int8_t *) chunk->data.get();
            n_heaps += n;
            double power = mean_power(adc_samples, present, heap_size, chunk_heaps);
            std::cout
                << "Timestamp: " << std::setw(10) << std::left << timestamp
                << " Power: " << power << '\n';
        }
        stream.add_free_chunk(std::move(chunk));
    }
    std::cout << "Received " << n_heaps << " heaps\n";
    return 0;
}
