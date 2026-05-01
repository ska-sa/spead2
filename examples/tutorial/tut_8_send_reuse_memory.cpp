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
#include <string>
#include <vector>
#include <array>
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
#include <spead2/common_thread_pool.h>
#include <spead2/send_heap.h>
#include <spead2/send_stream_config.h>
#include <spead2/send_udp.h>

struct state
{
    std::future<spead2::item_pointer_t> future;
    std::vector<std::int8_t> adc_samples;
    spead2::send::heap heap;

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
    config.set_max_heaps(2);
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

    std::array<state, 2> states;
    for (auto &state : states)
        state.adc_samples.resize(heap_size);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_heaps; i++)
    {
        auto &state = states[i % states.size()];
        // Wait for any previous use of this state to complete
        state.future.wait();
        auto &heap = state.heap;
        auto &adc_samples = state.adc_samples;

        heap = spead2::send::heap();  // reset to default state
        // Add descriptors to the first heap
        if (i == 0)
        {
            heap.add_descriptor(timestamp_desc);
            heap.add_descriptor(adc_samples_desc);
        }
        std::fill(adc_samples.begin(), adc_samples.end(), i);
        // Add the data and timestamp to the heap
        heap.add_item(timestamp_desc.id, i * heap_size);
        heap.add_item(
            adc_samples_desc.id,
            adc_samples.data(),
            adc_samples.size() * sizeof(adc_samples[0]),
            true
        );
        state.future = stream.async_send_heap(heap, boost::asio::use_future);
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
