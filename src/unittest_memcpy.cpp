#include <boost/test/unit_test.hpp>
#include <utility>
#include <cstdint>
#include <spead2/common_memcpy.h>

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(common)
BOOST_AUTO_TEST_SUITE(memcpy)

// Checks every combination of src and dest alignment relative to a page
BOOST_AUTO_TEST_CASE(memcpy_nontemporal_alignments)
{
    constexpr int head_pad = 32;
    constexpr int tail_pad = 32;
    constexpr int max_len = 128;
    constexpr int align_range = 64;
    constexpr int buffer_size = head_pad + align_range + max_len + tail_pad;

    std::uint8_t src_buffer[buffer_size];
    std::uint8_t dest_buffer[buffer_size];
    for (int i = 0; i < align_range; i++)
        for (int j = 0; j < align_range; j++)
            for (int len = 0; len <= max_len; len++)
            {
                std::memset(dest_buffer, 255, sizeof(dest_buffer));
                for (int k = 0; k < buffer_size; k++)
                    src_buffer[k] = k % 255;
                spead2::memcpy_nontemporal(dest_buffer + head_pad + i,
                                           src_buffer + head_pad + j, len);
                for (int k = 0; k < head_pad + i; k++)
                    BOOST_CHECK_EQUAL(255, dest_buffer[k]);
                for (int k = 0; k < len; k++)
                    BOOST_CHECK_EQUAL(src_buffer[head_pad + j + k], dest_buffer[head_pad + i + k]);
                for (int k = head_pad + i + len; k < buffer_size; k++)
                    BOOST_CHECK_EQUAL(255, dest_buffer[k]);
            }
}

BOOST_AUTO_TEST_SUITE_END()  // memcpy
BOOST_AUTO_TEST_SUITE_END()  // common

}} // namespace spead2::unittest
