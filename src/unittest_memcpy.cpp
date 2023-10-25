/* Copyright 2016, 2021, 2023 National Research Foundation (SARAO)
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
#include <boost/test/data/test_case.hpp>
#include <utility>
#include <cstdint>
#include <ostream>
#include <spead2/common_memcpy.h>

/* Declare the implementations of the instruction-specific implementations, so
 * that we can test all of them (that the current CPU supports) rather than
 * just the one selected by the resolver.
 */
namespace spead2
{
#if SPEAD2_USE_SSE2_STREAM
void *memcpy_nontemporal_sse2(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept;
#endif
#if SPEAD2_USE_AVX_STREAM
void *memcpy_nontemporal_avx(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept;
#endif
#if SPEAD2_USE_AVX512_STREAM
void *memcpy_nontemporal_avx512(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept;
#endif
} // namespace spead2

namespace spead2::unittest
{

BOOST_AUTO_TEST_SUITE(common)
BOOST_AUTO_TEST_SUITE(memcpy)

struct memcpy_function
{
    const char * name;
    void *(*func)(void * __restrict__, const void * __restrict__, std::size_t) noexcept;
    bool enabled;
};

std::ostream &operator<<(std::ostream &o, const memcpy_function &func)
{
    return o << func.name;
}

static const memcpy_function memcpy_functions[] =
{
    { "default", spead2::memcpy_nontemporal, true },
#if SPEAD2_USE_SSE2_STREAM
    { "sse2", spead2::memcpy_nontemporal_sse2, bool(__builtin_cpu_supports("sse2")) },
#endif
#if SPEAD2_USE_AVX_STREAM
    { "avx", spead2::memcpy_nontemporal_avx, bool(__builtin_cpu_supports("avx")) },
#endif
#if SPEAD2_USE_AVX512_STREAM
    { "avx512", spead2::memcpy_nontemporal_avx512, bool(__builtin_cpu_supports("avx512f")) },
#endif
};

// Checks combinations of src and dest alignment relative to a page
BOOST_DATA_TEST_CASE(memcpy_nontemporal_alignments, boost::unit_test::data::make(memcpy_functions), sample)
{
    if (!sample.enabled)
        return;

    constexpr int head_pad = 64;
    constexpr int tail_pad = 64;
    constexpr int max_len = 1024;
    constexpr int align_range = 64;
    constexpr int buffer_size = head_pad + align_range + max_len + tail_pad;

    std::uint8_t src_buffer[buffer_size];
    std::uint8_t dest_buffer[buffer_size];
    std::uint8_t expected[buffer_size];
    for (int i = 0; i < align_range; i += 3)
        for (int j = 0; j < align_range; j += 3)
            // Step 1 at a time up to 128, then take larger steps to reduce test time
            for (int len = 0; len <= max_len; len = (len < 128) ? len + 1 : len + 37)
            {
                std::memset(dest_buffer, 255, sizeof(dest_buffer));
                for (int k = 0; k < buffer_size; k++)
                    src_buffer[k] = k % 255;
                void *ret = sample.func(dest_buffer + head_pad + i, src_buffer + head_pad + j, len);
                BOOST_TEST(ret == dest_buffer + head_pad + i);

                std::memset(expected, 255, sizeof(expected));
                for (int k = 0; k < len; k++)
                    expected[head_pad + i + k] = src_buffer[head_pad + j + k];
                BOOST_TEST(dest_buffer == expected);
            }
}

BOOST_AUTO_TEST_SUITE_END()  // memcpy
BOOST_AUTO_TEST_SUITE_END()  // common

} // namespace spead2::unittest
