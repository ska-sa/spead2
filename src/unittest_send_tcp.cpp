/* Copyright 2020 National Research Foundation (SARAO)
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
 * Unit tests for send_tcp. This is just a targeted test for features that
 * aren't tested by the Python unit tests.
 */

#include <boost/asio.hpp>
#include <boost/test/unit_test.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/send_tcp.h>

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(send)
BOOST_AUTO_TEST_SUITE(tcp)

/* Put a heap into a stream before the connection has been established.
 * The connection is set up to fail, at which point we must get errors
 * reported.
 */
BOOST_AUTO_TEST_CASE(connect_fail)
{
    spead2::send::heap h;
    h.add_item(0x1234, 0x5678);

    spead2::thread_pool tp;
    boost::system::error_code connect_error;
    boost::system::error_code heap_error;

    boost::asio::ip::tcp::endpoint endpoint(
        boost::asio::ip::address_v4::from_string("127.0.0.1"),
        8887);
    spead2::send::tcp_stream stream(
        tp, [&](const boost::system::error_code &ec) { connect_error = ec; },
        {endpoint});
    auto handler = [&](const boost::system::error_code &ec, spead2::item_pointer_t bytes_transferred)
    {
        heap_error = ec;
    };
    stream.async_send_heap(h, handler);
    stream.flush();
    BOOST_CHECK_EQUAL(connect_error, boost::asio::error::connection_refused);
    BOOST_CHECK_EQUAL(heap_error, boost::asio::error::broken_pipe);
}

BOOST_AUTO_TEST_SUITE_END()  // tcp
BOOST_AUTO_TEST_SUITE_END()  // send

}} // namespace spead2::unittest
