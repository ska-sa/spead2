#include <boost/test/unit_test.hpp>
#include <utility>
#include <ostream>
#include <memory>
#include <spead2/common_memory_allocator.h>
#include <spead2/recv_live_heap.h>

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

BOOST_AUTO_TEST_CASE(payload_ranges)
{
    using spead2::recv::live_heap;
    live_heap heap(1, 0, std::make_shared<spead2::memory_allocator>());

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
