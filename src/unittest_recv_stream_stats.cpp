/* Copyright 2021 National Research Foundation (SARAO)
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
 * Unit tests for @ref spead2::recv::stream_stats and related classes. The
 * functionality is mostly tested via Python, but the Python bindings don't
 * allow all the C++ functionality to be exercised.
 */

#include <stdexcept>
#include <boost/test/unit_test.hpp>
#include <spead2/recv_stream.h>

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(recv)

BOOST_AUTO_TEST_SUITE(stream_stat_config)

BOOST_AUTO_TEST_CASE(test_equality)
{
    spead2::recv::stream_stat_config a("hello", spead2::recv::stream_stat_config::mode::COUNTER);
    spead2::recv::stream_stat_config b("hello", spead2::recv::stream_stat_config::mode::MAXIMUM);
    spead2::recv::stream_stat_config c("world", spead2::recv::stream_stat_config::mode::COUNTER);
    spead2::recv::stream_stat_config a2("hello");
    BOOST_CHECK(a != b);
    BOOST_CHECK(a != c);
    BOOST_CHECK(a == a2);
}

BOOST_AUTO_TEST_SUITE_END()  // stream_stat_config

BOOST_AUTO_TEST_SUITE(stream_stats)

BOOST_AUTO_TEST_CASE(test_iterator_equality)
{
    spead2::recv::stream_stats stats, stats2;
    BOOST_CHECK(stats.begin() != stats.end());
    BOOST_CHECK(stats.begin() != stats2.begin());
    BOOST_CHECK(stats.begin() == stats.begin());
}

BOOST_AUTO_TEST_CASE(test_iterator_compatibility)
{
    // Test compatibility between const_iterator and iterator
    spead2::recv::stream_stats stats;
    spead2::recv::stream_stats::const_iterator it{stats.begin()}; // constructor compatibility
    it = stats.begin();  // assignment compatibility
    BOOST_CHECK(it == stats.begin()); // comparison compatibility
}

BOOST_AUTO_TEST_CASE(test_iterator_traversal)
{
    spead2::recv::stream_stats stats;
    auto it = stats.begin();
    ++it;
    BOOST_CHECK(it == stats.begin() + 1);
    BOOST_CHECK(it - stats.begin() == 1);
    BOOST_CHECK(stats.begin() - it == -1);
    --it;
    BOOST_CHECK(it == stats.begin());
}

BOOST_AUTO_TEST_CASE(test_references)
{
    // Test that the backwards-compatibility fields work
    spead2::recv::stream_stats stats;
    stats["heaps"] = 1;
    stats["incomplete_heaps_evicted"] = 2;
    stats["incomplete_heaps_flushed"] = 3;
    stats["packets"] = 4;
    stats["batches"] = 5;
    stats["max_batch"] = 6;
    stats["single_packet_heaps"] = 7;
    stats["search_dist"] = 8;
    stats["worker_blocked"] = 9;
    BOOST_TEST(stats.heaps == 1);
    BOOST_TEST(stats.incomplete_heaps_evicted == 2);
    BOOST_TEST(stats.incomplete_heaps_flushed == 3);
    BOOST_TEST(stats.packets == 4);
    BOOST_TEST(stats.batches == 5);
    BOOST_TEST(stats.max_batch == 6);
    BOOST_TEST(stats.single_packet_heaps == 7);
    BOOST_TEST(stats.search_dist == 8);
    BOOST_TEST(stats.worker_blocked == 9);
};

BOOST_AUTO_TEST_CASE(test_index)
{
    spead2::recv::stream_stats stats;
    stats["heaps"] = 123;
    BOOST_TEST(stats["heaps"] == 123);
    BOOST_TEST(stats[spead2::recv::stream_stat_indices::heaps] == 123);
    BOOST_CHECK_THROW(stats["missing"], std::invalid_argument);

    const spead2::recv::stream_stats &stats_const = stats;
    BOOST_TEST(stats_const["heaps"] == 123);
    BOOST_TEST(stats_const[spead2::recv::stream_stat_indices::heaps] == 123);
    BOOST_CHECK_THROW(stats_const["missing"], std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()  // stream_stats
BOOST_AUTO_TEST_SUITE_END()  // recv

}} // namespace spead2::unittest
