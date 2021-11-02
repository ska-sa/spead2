/* Copyright 2016 National Research Foundation (SARAO)
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
 * Unit tests for accelerated memcpy.
 */

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <utility>
#include <random>
#include <cstdint>
#include <spead2/common_memcpy.h>

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(common)
BOOST_AUTO_TEST_SUITE(memcpy)

struct memcpy_nontemporal_wrapper
{
    void *operator()(void *dest, const void *src, std::size_t n)
    {
        return spead2::memcpy_nontemporal(dest, src, n);
    }
};

struct memcpy_nontemporal_rw_wrapper
{
    void *operator()(void *dest, const void *src, std::size_t n)
    {
        return spead2::memcpy_nontemporal_rw(dest, src, n);
    }
};

typedef boost::mpl::list<memcpy_nontemporal_wrapper, memcpy_nontemporal_rw_wrapper> memcpy_types;

// Checks every combination of src and dest alignment relative to a page
BOOST_AUTO_TEST_CASE_TEMPLATE(memcpy_nontemporal_alignments, T, memcpy_types)
{
    constexpr int head_pad = 32;
    constexpr int tail_pad = 32;
    constexpr int max_len = 128;
    constexpr int align_range = 64;
    constexpr int buffer_size = head_pad + align_range + max_len + tail_pad;
    std::mt19937 engine;
    std::uniform_int_distribution<std::uint8_t> dist(0, 254);

    std::uint8_t src_buffer[buffer_size];
    std::uint8_t dest_buffer[buffer_size];
    std::uint8_t expected[buffer_size];
    for (int i = 0; i < align_range; i++)
        for (int j = 0; j < align_range; j++)
            for (int len = 0; len <= max_len; len++)
            {
                std::memset(dest_buffer, 255, sizeof(dest_buffer));
                for (int k = 0; k < buffer_size; k++)
                    src_buffer[k] = dist(engine);
                T()(dest_buffer + head_pad + i, src_buffer + head_pad + j, len);

                std::memset(expected, 255, sizeof(expected));
                for (int k = 0; k < len; k++)
                    expected[head_pad + i + k] = src_buffer[head_pad + j + k];
                BOOST_TEST(dest_buffer == expected);
            }
}

BOOST_AUTO_TEST_SUITE_END()  // memcpy
BOOST_AUTO_TEST_SUITE_END()  // common

}} // namespace spead2::unittest
