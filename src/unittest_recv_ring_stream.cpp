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
 *
 * Unit tests for recv_ring_stream. It is mostly exercised via the Python API;
 * these tests are just for C++ features that aren't used by the Python
 * wrapper.
 */

#include <boost/asio.hpp>
#include <boost/test/unit_test.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/common_inproc.h>
#include <spead2/recv_inproc.h>
#include <spead2/recv_heap.h>
#include <spead2/recv_ring_stream.h>
#include <spead2/send_stream.h>
#include <spead2/send_inproc.h>
#include <spead2/send_heap.h>

namespace spead2::unittest
{

BOOST_AUTO_TEST_SUITE(recv)
BOOST_AUTO_TEST_SUITE(ring_stream)

// Test range-based for loop
BOOST_AUTO_TEST_CASE(iteration)
{
    thread_pool tp;
    auto queue = std::make_shared<inproc_queue>();
    // Feed some data into the inproc_queue
    spead2::send::inproc_stream send_stream(tp, {queue});
    for (int i = 0; i < 3; i++)
    {
        spead2::send::heap heap;
        heap.add_item(0x1000, i * 100);
        send_stream.async_send_heap(heap, boost::asio::use_future).wait();
    }
    queue->stop();

    // Receive the data, checking that the iterator receives the correct heaps
    spead2::recv::ring_stream<> recv_stream(tp);
    recv_stream.emplace_reader<spead2::recv::inproc_reader>(queue);
    std::vector<item_pointer_t> values;
    for (const spead2::recv::heap &heap : recv_stream)
    {
        for (auto &&item : heap.get_items())
            if (item.id == 0x1000)
                values.push_back(item.immediate_value);
    }

    std::vector<item_pointer_t> expected{0, 100, 200};
    BOOST_TEST(values == expected);
}

BOOST_AUTO_TEST_SUITE_END()  // ring_stream
BOOST_AUTO_TEST_SUITE_END()  // recv

} // namespace spead2::unittest
