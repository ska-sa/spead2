/* Copyright 2023, 2025 National Research Foundation (SARAO)
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
#include <random>
#include <string>
#include <vector>
#include <utility>
#include <boost/asio.hpp>
#include <spead2/common_defines.h>
#include <spead2/common_flavour.h>
#include <spead2/common_thread_pool.h>
#include <spead2/send_heap.h>
#include <spead2/send_stream_config.h>
#include <spead2/send_udp.h>

int main()
{
    spead2::thread_pool thread_pool;
    spead2::send::stream_config config;
    config.set_rate(100e6);
    boost::asio::ip::udp::endpoint endpoint(
        boost::asio::ip::make_address("127.0.0.1"),
        8888
    );
    spead2::send::udp_stream stream(thread_pool, {endpoint}, config);

    const std::int64_t heap_size = 1024 * 1024;
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

    std::default_random_engine random_engine;
    std::uniform_int_distribution<std::int8_t> distribution(-100, 100);
    std::vector<std::int8_t> adc_samples(heap_size);

    for (int i = 0; i < 10; i++)
    {
        spead2::send::heap heap;
        // Add descriptors to the first heap
        if (i == 0)
        {
            heap.add_descriptor(timestamp_desc);
            heap.add_descriptor(adc_samples_desc);
        }
        // Create random data
        for (int j = 0; j < heap_size; j++)
            adc_samples[j] = distribution(random_engine);
        // Add the data and timestamp to the heap
        heap.add_item(timestamp_desc.id, i * heap_size);
        heap.add_item(
            adc_samples_desc.id,
            adc_samples.data(),
            adc_samples.size() * sizeof(adc_samples[0]),
            true
        );
        stream.async_send_heap(heap, boost::asio::use_future).get();
    }

    // Send an end-of-stream control item
    spead2::send::heap heap;
    heap.add_end();
    stream.async_send_heap(heap, boost::asio::use_future).get();
    return 0;
}
