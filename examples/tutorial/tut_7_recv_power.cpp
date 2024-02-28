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

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <boost/asio.hpp>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_thread_pool.h>
#include <spead2/recv_ring_stream.h>
#include <spead2/recv_udp.h>
#include <spead2/recv_heap.h>

#if defined(__GNUC__) && defined(__x86_64__)
// Compile this function with AVX2 for better performance. Remove this if your
// CPU does not support AVX2 (e.g., if you get an Illegal Instruction error).
[[gnu::target("avx2")]]
#endif
static double mean_power(const std::int8_t *adc_samples, std::size_t length)
{
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < length; i++)
        sum += adc_samples[i] * adc_samples[i];
    return double(sum) / length;
}

int main()
{
    spead2::thread_pool thread_pool;
    spead2::recv::ring_stream stream(thread_pool);
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::any(), 8888);
    stream.emplace_reader<spead2::recv::udp_reader>(endpoint);
    std::int64_t n_heaps = 0;
    for (const spead2::recv::heap &heap : stream)
    {
        std::int64_t timestamp = -1;
        const std::int8_t *adc_samples = nullptr;
        std::size_t length = 0;
        for (const auto &item : heap.get_items())
        {
            if (item.id == 0x1600)
            {
                assert(item.is_immediate);
                timestamp = item.immediate_value;
            }
            else if (item.id == 0x3300)
            {
                adc_samples = reinterpret_cast<const std::int8_t *>(item.ptr);
                length = item.length;
            }
        }
        if (timestamp >= 0 && adc_samples != nullptr)
        {
            double power = mean_power(adc_samples, length);
            n_heaps++;
            std::cout
                << "Timestamp: " << std::setw(10) << std::left << timestamp
                << " Power: " << power << '\n';
        }
    }
    std::cout << "Received " << n_heaps << " heaps\n";
    return 0;
}
