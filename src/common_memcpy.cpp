/* Copyright 2016, 2020-2021 National Research Foundation (SARAO)
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
#include <spead2/common_defines.h>
#include <spead2/common_features.h>
#include <spead2/common_memcpy.h>
#if SPEAD2_USE_MOVNTDQ
# include <emmintrin.h>
# include <immintrin.h>
# include <tmmintrin.h>
# include <smmintrin.h>
#endif

namespace spead2
{

#if SPEAD2_USE_MOVNTDQ
[[gnu::target("sse2")]] static void *memcpy_nontemporal_sse2(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    // Align the destination to a cache-line boundary
    std::uintptr_t dest_i = std::uintptr_t(dest_c);
    constexpr std::uintptr_t cache_line_mask = detail::cache_line_size - 1;
    std::uintptr_t aligned = (dest_i + cache_line_mask) & ~cache_line_mask;
    std::size_t head = aligned - dest_i;
    if (head > 0)
    {
        if (head >= n)
        {
            std::memcpy(dest_c, src_c, n);
            /* Not normally required, but if the destination is
             * write-combining memory then this will flush the combining
             * buffers. That may be necessary if the memory is actually on
             * a GPU or other accelerator.
             */
            _mm_sfence();
            return dest;
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
}

[[gnu::target("avx")]] static void *memcpy_nontemporal_avx(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    // Align the destination to a cache-line boundary
    std::uintptr_t dest_i = std::uintptr_t(dest_c);
    constexpr std::uintptr_t cache_line_mask = detail::cache_line_size - 1;
    std::uintptr_t aligned = (dest_i + cache_line_mask) & ~cache_line_mask;
    std::size_t head = aligned - dest_i;
    if (head > 0)
    {
        if (head >= n)
        {
            std::memcpy(dest_c, src_c, n);
            /* Not normally required, but if the destination is
             * write-combining memory then this will flush the combining
             * buffers. That may be necessary if the memory is actually on
             * a GPU or other accelerator.
             */
            _mm_sfence();
            return dest;
        }
        std::memcpy(dest_c, src_c, head);
        dest_c += head;
        src_c += head;
        n -= head;
    }
    std::size_t offset;
    for (offset = 0; offset + 128 <= n; offset += 128)
    {
        __m256i value0 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 0));
        __m256i value1 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 32));
        __m256i value2 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 64));
        __m256i value3 = _mm256_loadu_si256((__m256i const *) (src_c + offset + 96));
        _mm256_stream_si256((__m256i *) (dest_c + offset + 0), value0);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 32), value1);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 64), value2);
        _mm256_stream_si256((__m256i *) (dest_c + offset + 96), value3);
    }
    _mm256_zeroupper();  // can apparently improve performance of subsequent SSE code
    std::size_t tail = n - offset;
    std::memcpy(dest_c + offset, src_c + offset, tail);
    _mm_sfence();
    return dest;
}
#endif // SPEAD2_USE_MOVNTDQ

[[gnu::target("default")]] void *memcpy_nontemporal(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    return memcpy(dest, src, n);
}

#if SPEAD2_USE_MOVNTDQ
[[gnu::target("sse2")]] void *memcpy_nontemporal(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    return memcpy_nontemporal_sse2(dest, src, n);
}

[[gnu::target("avx")]] void *memcpy_nontemporal(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    return memcpy_nontemporal_avx(dest, src, n);
}
#endif // SPEAD2_USE_MOVNTDQ

[[gnu::target("default")]] void *memcpy_nontemporal_rw(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    return memcpy_nontemporal(dest, src, n);
}

#if SPEAD2_USE_MOVNTDQ
template<int shift>
[[gnu::target("sse4.1")]] static void memcpy_nontemporal_rw_impl(__m128i * __restrict__ dest, const __m128i * __restrict__ src, std::size_t n) noexcept
{
    __m128i a = _mm_stream_load_si128(const_cast<__m128i *>(src));
    for (std::size_t i = 0; i < n; i++)
    {
        __m128i b = _mm_stream_load_si128(const_cast<__m128i *>(src + i + 1));
        __m128i out = _mm_alignr_epi8(b, a, shift);
        _mm_stream_si128(dest + i, out);
        a = b;
    }
}

[[gnu::target("sse4.1")]] void *memcpy_nontemporal_rw(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    constexpr std::uintptr_t cache_line_mask = detail::cache_line_size - 1;
    constexpr unsigned int vec_size = 16;

    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    std::uintptr_t src_i = std::uintptr_t(src_c);
    std::uintptr_t dest_i = std::uintptr_t(dest_c);
    unsigned int shift = (src_i - dest_i) & (vec_size - 1);

    // Align the destination to a cache-line boundary
    std::uintptr_t aligned = (dest_i + cache_line_mask) & ~cache_line_mask;
    std::size_t head = aligned - dest_i;
    if (head < shift)
        head += vec_size;  // ensure we don't read before the start
    // TODO: split this into an inline function
    if (head + vec_size >= n)
    {
        std::memcpy(dest_c, src_c, n);
        /* Not normally required, but if the destination is
         * write-combining memory then this will flush the combining
         * buffers. That may be necessary if the memory is actually on
         * a GPU or other accelerator.
         */
        _mm_sfence();
        return dest;
    }
    std::memcpy(dest_c, src_c, head);
    dest_c += head;
    src_c += head;
    n -= head;

    // The - 1 is to ensure that loads don't go over the end of the source.
    // It could probably be squeezed a bit further by considering the shift
    // and the size of the tail.
    std::size_t vecs = n / vec_size - 1;
#define HANDLE_SHIFT(s) \
    case s: memcpy_nontemporal_rw_impl<s>( \
        (__m128i *) dest_c, (const __m128i *) (src_c - shift), vecs); break
    switch (shift)
    {
        HANDLE_SHIFT(0);
        HANDLE_SHIFT(1);
        HANDLE_SHIFT(2);
        HANDLE_SHIFT(3);
        HANDLE_SHIFT(4);
        HANDLE_SHIFT(5);
        HANDLE_SHIFT(6);
        HANDLE_SHIFT(7);
        HANDLE_SHIFT(8);
        HANDLE_SHIFT(9);
        HANDLE_SHIFT(10);
        HANDLE_SHIFT(11);
        HANDLE_SHIFT(12);
        HANDLE_SHIFT(13);
        HANDLE_SHIFT(14);
        HANDLE_SHIFT(15);
    };
#undef HANDLE_SHIFT

    std::size_t done = vecs * vec_size;
    src_c += done;
    dest_c += done;
    n -= done;
    std::memcpy(dest_c, src_c, n);
    _mm_sfence();
    return dest;
}
#endif // SPEAD2_USE_MOVNTDQ

} // namespace spead2
