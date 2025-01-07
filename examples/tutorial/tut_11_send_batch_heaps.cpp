/* Copyright 2023-2025 National Research Foundation (SARAO)
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
#include <string>
#include <vector>
#include <utility>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <future>
#include <optional>
#include <unistd.h>
#include <boost/asio.hpp>
#include <spead2/common_defines.h>
#include <spead2/common_flavour.h>
#include <spead2/common_endian.h>
#include <spead2/common_thread_pool.h>
#include <spead2/send_heap.h>
#include <spead2/send_stream_config.h>
#include <spead2/send_udp.h>

struct heap_data
{
    std::vector<std::int8_t> adc_samples;
    std::uint64_t timestamp;  // big endian
};

struct state
{
    std::future<spead2::item_pointer_t> future;
    std::vector<heap_data> data;
    std::vector<spead2::send::heap> heaps;

    state()
    {
        // Make it safe to wait on the future immediately
        std::promise<spead2::item_pointer_t> promise;
        promise.set_value(0);
        future = promise.get_future();
    }
};

static void usage(const char * name)
{
    std::cerr << "Usage: " << name << " [-n heaps] [-p packet-size] [-H heap-size] host port\n";
}

int main(int argc, char * const argv[])
{
    int opt;
    int n_heaps = 1000;
    std::optional<int> packet_size;
    std::int64_t heap_size = 1024 * 1024;
    while ((opt = getopt(argc, argv, "n:p:H:")) != -1)
    {
        switch (opt)
        {
        case 'n':
            n_heaps = std::stoi(optarg);
            break;
        case 'p':
            packet_size = std::stoi(optarg);
            break;
        case 'H':
            heap_size = std::stoll(optarg);
            break;
        default:
            usage(argv[0]);
            return 2;
        }
    }
    if (argc - optind != 2)
    {
        usage(argv[0]);
        return 2;
    }

    spead2::thread_pool thread_pool(1, {0});
    spead2::thread_pool::set_affinity(1);
    spead2::send::stream_config config;
    config.set_rate(0.0);
    const std::size_t batches = 2;
    const std::size_t batch_heaps = std::max(std::int64_t(2), 512 * 1024 / heap_size);
    config.set_max_heaps(batches * batch_heaps);
    if (packet_size)
        config.set_max_packet_size(packet_size.value());
    boost::asio::ip::udp::endpoint endpoint(
        boost::asio::ip::make_address(argv[optind]),
        std::stoi(argv[optind + 1])
    );
    spead2::send::udp_stream stream(thread_pool, {endpoint}, config);

    spead2::descriptor timestamp_desc;
    timestamp_desc.id = 0x1600;
    timestamp_desc.name = "timestamp";
    timestamp_desc.description = "Index of the first sample";
    timestamp_desc.format.emplace_back('u', spead2::flavour().get_heap_address_bits());
    spead2::descriptor adc_samples_desc;
    adc_samples_desc.id = 0x3300;
    adc_samples_desc.name = "adc_samples";
    adc_samples_desc.description = "ADC converter output";
    adc_samples_desc.numpy_header =
        "{'shape': (" + std::to_string(heap_size) + ",), 'fortran_order': False, 'descr': 'i1'}";

    auto make_heaps = [&](const std::vector<heap_data> &data, bool first)
    {
        std::vector<spead2::send::heap> heaps(batch_heaps);
        for (std::size_t j = 0; j < batch_heaps; j++)
        {
            auto &heap = heaps[j];
            auto &adc_samples = data[j].adc_samples;
            auto &timestamp = data[j].timestamp;
            if (first && j == 0)
            {
                heap.add_descriptor(timestamp_desc);
                heap.add_descriptor(adc_samples_desc);
            }
            heap.add_item(timestamp_desc.id, (char *) &timestamp + 3, 5, true);
            heap.add_item(
                adc_samples_desc.id,
                adc_samples.data(),
                adc_samples.size() * sizeof(adc_samples[0]),
                true
            );
        }
        return heaps;
    };

    std::vector<state> states(batches);
    for (std::size_t i = 0; i < states.size(); i++)
    {
        auto &state = states[i];
        state.data.resize(batch_heaps);
        for (std::size_t j = 0; j < batch_heaps; j++)
        {
            state.data[j].adc_samples.resize(heap_size);
        }
        state.heaps = make_heaps(state.data, false);
    }
    auto first_heaps = make_heaps(states[0].data, true);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_heaps; i += batch_heaps)
    {
        auto &state = states[(i / batch_heaps) % states.size()];
        // Wait for any previous use of this state to complete
        state.future.wait();
        auto &heaps = (i == 0) ? first_heaps : state.heaps;
        std::int64_t end = std::min(n_heaps, i + int(batch_heaps));
        std::size_t n = end - i;
        for (std::size_t j = 0; j < n; j++)
        {
            std::int64_t heap_index = i + j;
            auto &data = state.data[j];
            auto &adc_samples = data.adc_samples;
            data.timestamp = spead2::htobe(std::uint64_t(heap_index * heap_size));
            std::fill(adc_samples.begin(), adc_samples.end(), heap_index);
        }
        state.future = stream.async_send_heaps(
            heaps.begin(), heaps.begin() + n, boost::asio::use_future,
            spead2::send::group_mode::SERIAL
        );
    }
    for (const auto &state : states)
        state.future.wait();
    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    std::cout << heap_size * n_heaps / elapsed.count() / 1e6 << " MB/s\n";

    // Send an end-of-stream control item
    spead2::send::heap heap;
    heap.add_end();
    stream.async_send_heap(heap, boost::asio::use_future).get();
    return 0;
}
