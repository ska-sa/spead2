/* Copyright 2019 SKA South Africa
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
 * Unit tests for recv stream with custom memcpy function.
 */

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <boost/test/unit_test.hpp>
#include <spead2/recv_stream.h>
#include <spead2/recv_ring_stream.h>
#include <spead2/recv_inproc.h>
#include <spead2/recv_packet.h>
#include <spead2/send_stream.h>
#include <spead2/send_heap.h>
#include <spead2/send_inproc.h>
#include <spead2/common_inproc.h>

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(recv)
BOOST_AUTO_TEST_SUITE(custom_memcpy)

static void reverse_memcpy(const spead2::memory_allocator::pointer &allocation, const spead2::recv::packet_header &packet)
{
    std::uint8_t *ptr = allocation.get() + (packet.heap_length - packet.payload_offset);
    for (std::size_t i = 0; i < packet.payload_length; i++)
        *--ptr = packet.payload[i];
}

/* Set up a receive stream that uses a custom memcpy to
 * reverse all the payload bytes in a heap.
 */
BOOST_AUTO_TEST_CASE(test_reverse)
{
    // Create some random data to transmit
    std::mt19937 engine;
    std::uniform_int_distribution<int> bytes_dist(0, 255);
    std::vector<std::uint8_t> data(100000);
    for (auto &v : data)
        v = bytes_dist(engine);

    // Set up receiver
    thread_pool tp;
    std::shared_ptr<inproc_queue> queue = std::make_shared<inproc_queue>();
    spead2::recv::ring_stream<> recv_stream(tp);
    recv_stream.set_allow_unsized_heaps(false);
    recv_stream.set_memcpy(reverse_memcpy);
    recv_stream.emplace_reader<spead2::recv::inproc_reader>(queue);

    // Set up sender and send the heap
    spead2::send::inproc_stream send_stream(tp, queue);
    flavour f(4, 64, 48);
    spead2::send::heap send_heap(f);
    spead2::send::heap stop_heap(f);
    send_heap.add_item(0x1000, data.data(), data.size(), false);
    stop_heap.add_end();
    send_stream.async_send_heap(
        send_heap,
        [&](const boost::system::error_code &ec, item_pointer_t bytes_transferred) {});
    send_stream.async_send_heap(
        stop_heap,
        [&](const boost::system::error_code &ec, item_pointer_t bytes_transferred) {});
    send_stream.flush();

    // Retrieve the heap and check the content
    spead2::recv::heap recv_heap = recv_stream.pop();
    bool found = false;
    for (const auto &item : recv_heap.get_items())
    {
        if (item.id == 0x1000)
        {
            BOOST_CHECK(!found);
            BOOST_CHECK_EQUAL_COLLECTIONS(data.rbegin(), data.rend(), item.ptr, item.ptr + item.length);
            found = true;
        }
    }
    BOOST_CHECK(found);
}

BOOST_AUTO_TEST_SUITE_END()  // custom_memcpy
BOOST_AUTO_TEST_SUITE_END()  // recv

}} // namespace spead2::unittest
