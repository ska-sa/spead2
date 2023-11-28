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
 * Unit tests for send completion tokens.
 */

#include <exception>
#include <memory>
#include <future>
#include <utility>
#include <cstddef>
#include <boost/asio.hpp>
#include <boost/test/unit_test.hpp>
#include <spead2/common_defines.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_inproc.h>
#include <spead2/send_inproc.h>

namespace spead2::unittest
{

BOOST_AUTO_TEST_SUITE(send)
BOOST_AUTO_TEST_SUITE(completion)

// empty heap: header, 4 standard items, padding item, padding
static constexpr std::size_t heap_size = 9 + 5 * sizeof(item_pointer_t);

class promise_handler
{
    std::promise<item_pointer_t> &promise;

public:
    explicit promise_handler(std::promise<item_pointer_t> &promise) : promise(promise) {}

    void operator()(const boost::system::error_code &ec, item_pointer_t bytes_transferred) const
    {
        if (ec)
            promise.set_exception(std::make_exception_ptr(boost::system::system_error(ec)));
        else
            promise.set_value(bytes_transferred);
    }
};

static bool is_would_block(const boost::system::system_error &ex)
{
    return ex.code() == boost::asio::error::would_block;
}

// Test async_send_heap with a completion handler
BOOST_AUTO_TEST_CASE(async_send_heap_handler)
{
    thread_pool tp;
    auto queue = std::make_shared<inproc_queue>();
    spead2::send::inproc_stream stream(tp, {queue});
    spead2::send::heap heap;

    std::promise<item_pointer_t> promise;
    auto future = promise.get_future();
    bool result = stream.async_send_heap(heap, promise_handler(promise));
    BOOST_CHECK_EQUAL(result, true);
    BOOST_CHECK_EQUAL(future.get(), heap_size);
}

// Test async_send_heap with a generic token
BOOST_AUTO_TEST_CASE(async_send_heap_token)
{
    thread_pool tp;
    auto queue = std::make_shared<inproc_queue>();
    spead2::send::inproc_stream stream(tp, {queue});
    spead2::send::heap heap;

    std::future<item_pointer_t> future = stream.async_send_heap(heap, boost::asio::use_future);
    BOOST_CHECK_EQUAL(future.get(), heap_size);
}

// Test async_send_heaps with a completion handler
BOOST_AUTO_TEST_CASE(async_send_heaps_handler)
{
    thread_pool tp;
    auto queue = std::make_shared<inproc_queue>();
    spead2::send::inproc_stream stream(tp, {queue});
    std::array<spead2::send::heap, 2> heaps;

    std::promise<item_pointer_t> promise;
    auto future = promise.get_future();
    bool result = stream.async_send_heaps(
        heaps.begin(), heaps.end(),
        promise_handler(promise),
        spead2::send::group_mode::SERIAL
    );
    BOOST_CHECK_EQUAL(result, true);
    BOOST_CHECK_EQUAL(future.get(), heap_size * heaps.size());
}

// Test async_send_heaps with a completion token
BOOST_AUTO_TEST_CASE(async_send_heaps_token)
{
    thread_pool tp;
    auto queue = std::make_shared<inproc_queue>();
    spead2::send::inproc_stream stream(tp, {queue});
    std::array<spead2::send::heap, 2> heaps;

    std::future<item_pointer_t> future = stream.async_send_heaps(
        heaps.begin(), heaps.end(),
        boost::asio::use_future,
        spead2::send::group_mode::SERIAL
    );
    BOOST_CHECK_EQUAL(future.get(), heap_size * heaps.size());
}

// Test async_send_heaps failure case with a completion handler
BOOST_AUTO_TEST_CASE(async_send_heaps_failure_handler)
{
    thread_pool tp;
    auto queue = std::make_shared<inproc_queue>();
    spead2::send::inproc_stream stream(tp, {queue}, spead2::send::stream_config().set_max_heaps(1));
    std::array<spead2::send::heap, 2> heaps;

    std::promise<item_pointer_t> promise;
    auto future = promise.get_future();
    bool result = stream.async_send_heaps(
        heaps.begin(), heaps.end(),
        promise_handler(promise),
        spead2::send::group_mode::SERIAL
    );
    BOOST_CHECK_EQUAL(result, false);
    BOOST_CHECK_EXCEPTION(future.get(), boost::system::system_error, is_would_block);
}

// Test async_send_heaps failure case with a completion token
BOOST_AUTO_TEST_CASE(async_send_heaps_failure_token)
{
    thread_pool tp;
    auto queue = std::make_shared<inproc_queue>();
    spead2::send::inproc_stream stream(tp, {queue}, spead2::send::stream_config().set_max_heaps(1));
    std::array<spead2::send::heap, 2> heaps;

    std::promise<item_pointer_t> promise;
    auto future = promise.get_future();
    bool result = stream.async_send_heaps(
        heaps.begin(), heaps.end(),
        [&promise](const boost::system::error_code &ec, item_pointer_t bytes_transferred)
        {
            if (ec)
                promise.set_exception(std::make_exception_ptr(boost::system::system_error(ec)));
            else
                promise.set_value(bytes_transferred);
        },
        spead2::send::group_mode::SERIAL
    );
    BOOST_CHECK_EQUAL(result, false);
    BOOST_CHECK_EXCEPTION(future.get(), boost::system::system_error, is_would_block);
}

BOOST_AUTO_TEST_SUITE_END()  // completion
BOOST_AUTO_TEST_SUITE_END()  // send

} // namespace spead2::unittest
