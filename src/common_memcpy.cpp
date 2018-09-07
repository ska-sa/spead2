/* Copyright 2016 SKA South Africa
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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <spead2/common_features.h>
#include <spead2/common_memcpy.h>
#if SPEAD2_USE_MOVNTDQ
# include <emmintrin.h>
#endif

namespace spead2
{

void *memcpy_nontemporal(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
#if !SPEAD2_USE_MOVNTDQ
    return std::memcpy(dest, src, n);
#else
    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    // Align the destination to a cache-line boundary, assuming 64-byte cache lines
    std::uintptr_t dest_i = std::uintptr_t(dest_c);
    std::uintptr_t aligned = (dest_i + 63) & ~63;
    std::size_t head = aligned - dest_i;
    if (head > 0)
    {
        if (head >= n)
        {
            return std::memcpy(dest_c, src_c, n);
        }
        std::memcpy(dest_c, src_c, head);
        dest_c += head;
        src_c += head;
        n -= head;
    }
    std::size_t offset;
    for (offset = 0; offset + 64 <= n; offset += 64)
    {
        __m128i value0 = _mm_loadu_si128((__m128i const *) (src_c + offset + 0));
        __m128i value1 = _mm_loadu_si128((__m128i const *) (src_c + offset + 16));
        __m128i value2 = _mm_loadu_si128((__m128i const *) (src_c + offset + 32));
        __m128i value3 = _mm_loadu_si128((__m128i const *) (src_c + offset + 48));
        _mm_stream_si128((__m128i *) (dest_c + offset + 0), value0);
        _mm_stream_si128((__m128i *) (dest_c + offset + 16), value1);
        _mm_stream_si128((__m128i *) (dest_c + offset + 32), value2);
        _mm_stream_si128((__m128i *) (dest_c + offset + 48), value3);
    }
    std::size_t tail = n - offset;
    std::memcpy(dest_c + offset, src_c + offset, tail);
    _mm_sfence();
    return dest;
#endif // SPEAD2_USE_MOVNTDQ
}

} // namespace spead2
