/* Copyright 2016, 2018 SKA South Africa
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
 * Unit tests for recv_live_heap.
 */

#include <boost/test/unit_test.hpp>
#include <utility>
#include <ostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_packet.h>
#include <spead2/common_endian.h>

namespace std
{

template<typename A, typename B>
ostream &operator<<(ostream &o, const pair<A, B> &v)
{
    return o << "(" << v.first << ", " << v.second << ")";
}

} // namespace std

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(recv)
BOOST_AUTO_TEST_SUITE(live_heap)

// Creates a packet_header with just enough in it to initialise a live_heap
static spead2::recv::packet_header dummy_packet(
    s_item_pointer_t heap_cnt, int heap_address_bits = 48)
{
    spead2::recv::packet_header header;
    header.heap_address_bits = heap_address_bits;
    header.n_items = 0;
    header.heap_cnt = heap_cnt;
    header.heap_length = 0;
    header.payload_offset = 0;
    header.payload_length = 0;
    header.pointers = NULL;
    header.payload = NULL;
    return header;
}

BOOST_AUTO_TEST_CASE(add_pointers)
{
    using spead2::recv::live_heap;
    live_heap heap(dummy_packet(1), 0);

    std::vector<item_pointer_t> in_pointers[3]{
        {
            0x0005000000000000,
            0x0005000000001000,
            0x0005000000000000,
            0x9234000000001234,
            0x9235AAAAAAAAAAAA
        },
        {},
        {
            0x0005000000004321,
            0x9235AAAAAAAAAAAA
        }
    };

    for (int i = 0; i < 30; i++)
        in_pointers[1].push_back(item_pointer_t(0x9000 + i) << 48);
    in_pointers[1].push_back(0x9234000000001234);

    std::vector<item_pointer_t> expected;
    for (auto &in : in_pointers)
    {
        // Update the expected results
        for (const auto &pointer : in)
            if (!std::count(expected.begin(), expected.end(), pointer))
                expected.push_back(pointer);
        // Convert the pointers to big endian, which add_pointers expects
        for (auto &pointer : in)
            pointer = htobe(pointer);
        heap.add_pointers(in.size(),
                          reinterpret_cast<const std::uint8_t *>(in.data()));
        BOOST_CHECK_EQUAL_COLLECTIONS(heap.pointers_begin(), heap.pointers_end(),
                                      expected.begin(), expected.end());
    }
}

BOOST_AUTO_TEST_CASE(payload_ranges)
{
    using spead2::recv::live_heap;
    live_heap heap(dummy_packet(1), 0);

    BOOST_CHECK(heap.add_payload_range(100, 200));
    std::pair<const s_item_pointer_t, s_item_pointer_t> expected1[] = {{100, 200}};
    BOOST_CHECK_EQUAL_COLLECTIONS(heap.payload_ranges.begin(), heap.payload_ranges.end(),
                                  std::begin(expected1), std::end(expected1));
    // Range prior to all previous values
    BOOST_CHECK(heap.add_payload_range(30, 40));
    std::pair<const s_item_pointer_t, s_item_pointer_t> expected2[] = {{30, 40}, {100, 200}};
    BOOST_CHECK_EQUAL_COLLECTIONS(heap.payload_ranges.begin(), heap.payload_ranges.end(),
                                  std::begin(expected2), std::end(expected2));
    // Range after all existing values
    BOOST_CHECK(heap.add_payload_range(300, 350));
    std::pair<const s_item_pointer_t, s_item_pointer_t> expected3[] = {{30, 40}, {100, 200}, {300, 350}};
    BOOST_CHECK_EQUAL_COLLECTIONS(heap.payload_ranges.begin(), heap.payload_ranges.end(),
                                  std::begin(expected3), std::end(expected3));
    // Insert at start, merge right
    BOOST_CHECK(heap.add_payload_range(25, 30));
    std::pair<const s_item_pointer_t, s_item_pointer_t> expected4[] = {{25, 40}, {100, 200}, {300, 350}};
    BOOST_CHECK_EQUAL_COLLECTIONS(heap.payload_ranges.begin(), heap.payload_ranges.end(),
                                  std::begin(expected4), std::end(expected4));
    // Insert at end, merge left
    BOOST_CHECK(heap.add_payload_range(350, 360));
    std::pair<const s_item_pointer_t, s_item_pointer_t> expected5[] = {{25, 40}, {100, 200}, {300, 360}};
    BOOST_CHECK_EQUAL_COLLECTIONS(heap.payload_ranges.begin(), heap.payload_ranges.end(),
                                  std::begin(expected5), std::end(expected5));
    // Insert in middle, merge left
    BOOST_CHECK(heap.add_payload_range(40, 50));
    std::pair<const s_item_pointer_t, s_item_pointer_t> expected6[] = {{25, 50}, {100, 200}, {300, 360}};
    BOOST_CHECK_EQUAL_COLLECTIONS(heap.payload_ranges.begin(), heap.payload_ranges.end(),
                                  std::begin(expected6), std::end(expected6));
    // Insert in middle, merge right
    BOOST_CHECK(heap.add_payload_range(80, 100));
    std::pair<const s_item_pointer_t, s_item_pointer_t> expected7[] = {{25, 50}, {80, 200}, {300, 360}};
    BOOST_CHECK_EQUAL_COLLECTIONS(heap.payload_ranges.begin(), heap.payload_ranges.end(),
                                  std::begin(expected7), std::end(expected7));
    // Insert in middle, no merge
    BOOST_CHECK(heap.add_payload_range(60, 70));
    std::pair<const s_item_pointer_t, s_item_pointer_t> expected8[] = {{25, 50}, {60, 70}, {80, 200}, {300, 360}};
    BOOST_CHECK_EQUAL_COLLECTIONS(heap.payload_ranges.begin(), heap.payload_ranges.end(),
                                  std::begin(expected8), std::end(expected8));
    // Insert in middle, merge both sides
    BOOST_CHECK(heap.add_payload_range(50, 60));
    std::pair<const s_item_pointer_t, s_item_pointer_t> expected9[] = {{25, 70}, {80, 200}, {300, 360}};
    BOOST_CHECK_EQUAL_COLLECTIONS(heap.payload_ranges.begin(), heap.payload_ranges.end(),
                                  std::begin(expected9), std::end(expected9));
    // Duplicates of various sorts
    BOOST_CHECK(!heap.add_payload_range(40, 50));
    BOOST_CHECK(!heap.add_payload_range(25, 30));
    BOOST_CHECK(!heap.add_payload_range(90, 200));
    BOOST_CHECK(!heap.add_payload_range(300, 360));
}

BOOST_AUTO_TEST_SUITE_END()  // live_heap
BOOST_AUTO_TEST_SUITE_END()  // recv

}} // namespace spead2::unittest
