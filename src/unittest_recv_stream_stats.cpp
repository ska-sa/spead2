/* Copyright 2021, 2023 National Research Foundation (SARAO)
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

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <boost/test/unit_test.hpp>
#include <spead2/recv_stream.h>

namespace spead2::unittest
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

BOOST_AUTO_TEST_CASE(test_reverse_iteration)
{
    spead2::recv::stream_stats stats;
    std::vector<std::string> names, rnames, crnames;
    /* Uses (*it).first rather than it->first because libc++ doesn't like the
     * latter with reverse iterators when dereferencing doesn't return an
     * lvalue.
     */
    for (auto it = stats.begin(); it != stats.end(); ++it)
        names.push_back((*it).first);
    for (auto it = stats.rbegin(); it != stats.rend(); ++it)
        rnames.push_back((*it).first);
    for (auto it = stats.crbegin(); it != stats.crend(); ++it)
        crnames.push_back((*it).first);
    std::reverse(rnames.begin(), rnames.end());
    BOOST_TEST(rnames == names);
    std::reverse(crnames.begin(), crnames.end());
    BOOST_TEST(crnames == names);
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
    BOOST_TEST(stats.at("heaps") == 123);
    BOOST_TEST(stats[spead2::recv::stream_stat_indices::heaps] == 123);
    BOOST_CHECK_THROW(stats["missing"], std::out_of_range);
    BOOST_CHECK_THROW(stats.at("missing"), std::out_of_range);
    BOOST_CHECK_THROW(stats.at(100000), std::out_of_range);

    const spead2::recv::stream_stats &stats_const = stats;
    BOOST_TEST(stats_const["heaps"] == 123);
    BOOST_TEST(stats_const.at("heaps") == 123);
    BOOST_TEST(stats_const[spead2::recv::stream_stat_indices::heaps] == 123);
    BOOST_CHECK_THROW(stats_const["missing"], std::out_of_range);
    BOOST_CHECK_THROW(stats_const.at("missing"), std::out_of_range);
    BOOST_CHECK_THROW(stats_const.at(100000), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(test_find)
{
    spead2::recv::stream_stats stats;
    stats.at("heaps") = 123;
    auto it = stats.find("heaps");
    BOOST_REQUIRE(it != stats.end());
    BOOST_TEST(it->first == "heaps");
    BOOST_TEST(it->second == 123);
    it = stats.find("missing");
    BOOST_CHECK(it == stats.end());

    const spead2::recv::stream_stats &stats_const = stats;
    auto it_const = stats_const.find("heaps");
    BOOST_REQUIRE(it_const != stats_const.end());
    BOOST_TEST(it_const->first == "heaps");
    BOOST_TEST(it_const->second == 123);
    it_const = stats_const.find("missing");
    BOOST_CHECK(it_const == stats_const.end());
}

BOOST_AUTO_TEST_CASE(test_count)
{
    spead2::recv::stream_stats stats;
    BOOST_TEST(stats.count("heaps") == 1);
    BOOST_TEST(stats.count("missing") == 0);
}

BOOST_AUTO_TEST_CASE(test_copy)
{
    spead2::recv::stream_stats stats1;
    stats1.packets = 10;
    spead2::recv::stream_stats stats2(stats1);
    BOOST_TEST(&stats2.packets == &stats2["packets"]);
    stats1.packets = 20;
    BOOST_TEST(stats2.packets == 10);
}

BOOST_AUTO_TEST_CASE(test_copy_assign)
{
    spead2::recv::stream_stats stats1, stats2;
    stats1.packets = 10;
    stats2 = stats1;
    BOOST_TEST(stats2.packets == 10);
    BOOST_TEST(&stats2.packets == &stats2["packets"]);
}

BOOST_AUTO_TEST_CASE(test_move)
{
    spead2::recv::stream_stats stats1;
    stats1.packets = 10;
    spead2::recv::stream_stats stats2(std::move(stats1));
    BOOST_TEST(&stats2.packets == &stats2["packets"]);
    BOOST_TEST(stats2.packets == 10);
}

BOOST_AUTO_TEST_CASE(test_move_assign)
{
    spead2::recv::stream_stats stats1, stats2;
    stats1.packets = 10;
    stats2 = std::move(stats1);
    BOOST_TEST(stats2.packets == 10);
    BOOST_TEST(&stats2.packets == &stats2["packets"]);
}

BOOST_AUTO_TEST_SUITE_END()  // stream_stats
BOOST_AUTO_TEST_SUITE_END()  // recv

} // namespace spead2::unittest
