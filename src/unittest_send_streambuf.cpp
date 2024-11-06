/* Copyright 2019-2020, 2023-2024 National Research Foundation (SARAO)
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
 * Unit tests for send_streambuf.
 */

#include <array>
#include <utility>
#include <boost/test/unit_test.hpp>
#include <spead2/send_streambuf.h>

namespace spead2::unittest
{

BOOST_AUTO_TEST_SUITE(send)
BOOST_AUTO_TEST_SUITE(streambuf)

namespace
{

/* streambuf that does not implement overflow(). It will thus report EOF as
 * soon as it runs out of its pre-defined buffer space.
 */
class array_streambuf : public std::streambuf
{
public:
    array_streambuf(char *first, char *last)
    {
        setp(first, last);
    }
};

}   // anonymous namespace

/* Send data to a streambuf that returns an EOF error
 * (the success case is tested more thoroughly via Python).
 */
BOOST_AUTO_TEST_CASE(send_fail)
{
    spead2::thread_pool tp;
    spead2::send::heap h;
    h.add_item(0x1234, 0x5678);
    std::array<char, 20> buffer;
    array_streambuf sb(buffer.begin(), buffer.end());
    spead2::send::streambuf_stream stream(tp, sb);
    std::promise<std::pair<boost::system::error_code, std::size_t>> result_promise;
    auto handler = [&](const boost::system::error_code &ec, std::size_t bytes_transferred)
    {
        result_promise.set_value(std::pair(ec, bytes_transferred));
    };
    stream.async_send_heap(h, handler);
    auto [ec, bytes_transferred] = result_promise.get_future().get();
    BOOST_TEST(ec == boost::asio::error::eof);
    BOOST_TEST(bytes_transferred == 20U);
}

BOOST_AUTO_TEST_SUITE_END()  // streambuf
BOOST_AUTO_TEST_SUITE_END()  // send

} // namespace spead2::unittest
