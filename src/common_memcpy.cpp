/* Copyright 2016, 2020, 2023 National Research Foundation (SARAO)
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
#include <utility>
#include <spead2/common_defines.h>
#include <spead2/common_features.h>
#include <spead2/common_memcpy.h>

#if SPEAD2_USE_SSE2_STREAM
# include <emmintrin.h>
# define SPEAD2_MEMCPY_NAME memcpy_nontemporal_sse2
# define SPEAD2_MEMCPY_TARGET "sse2"
# define SPEAD2_MEMCPY_TYPE __m128i
# define SPEAD2_MEMCPY_LOAD _mm_loadu_si128
# define SPEAD2_MEMCPY_STORE _mm_stream_si128
# define SPEAD2_MEMCPY_UNROLL 16
# define SPEAD2_MEMCPY_VZEROUPPER 0
# include "common_memcpy_impl.h"
#endif

#if SPEAD2_USE_AVX_STREAM
# include <immintrin.h>
# define SPEAD2_MEMCPY_NAME memcpy_nontemporal_avx
# define SPEAD2_MEMCPY_TARGET "avx"
# define SPEAD2_MEMCPY_TYPE __m256i
# define SPEAD2_MEMCPY_LOAD _mm256_loadu_si256
# define SPEAD2_MEMCPY_STORE _mm256_stream_si256
# define SPEAD2_MEMCPY_UNROLL 8
# define SPEAD2_MEMCPY_VZEROUPPER 1
# include "common_memcpy_impl.h"
#endif

#if SPEAD2_USE_AVX512_STREAM
# include <immintrin.h>
# define SPEAD2_MEMCPY_NAME memcpy_nontemporal_avx512
# define SPEAD2_MEMCPY_TARGET "avx512f"
# define SPEAD2_MEMCPY_TYPE __m512i
# define SPEAD2_MEMCPY_LOAD _mm512_loadu_si512
# define SPEAD2_MEMCPY_STORE _mm512_stream_si512
# define SPEAD2_MEMCPY_UNROLL 8
# define SPEAD2_MEMCPY_VZEROUPPER 1
# include "common_memcpy_impl.h"
#endif

namespace spead2
{

void *(*resolve_memcpy_nontemporal())(void *, const void *, std::size_t) noexcept
{
#if SPEAD2_USE_AVX512_STREAM || SPEAD2_USE_AVX_STREAM || SPEAD2_USE_SSE2_STREAM
    __builtin_cpu_init();
#endif
#if SPEAD2_USE_AVX512_STREAM
    /* On Skylake server, AVX-512 reduces clock speeds. Use the same logic as
     * Glibc to decide whether AVX-512 is okay: it's okay if either AVX512ER or
     * AVX512-VNNI is present. Glibc only applies that logic to Intel CPUs, but
     * AMD introduced AVX-512 with Zen 4 which also supports AVX512-VNNI (and
     * performs well), so we don't need to distinguish.
     */
    if (__builtin_cpu_supports("avx512f")
        && (__builtin_cpu_supports("avx512er") || __builtin_cpu_supports("avx512vnni")))
        return memcpy_nontemporal_avx512;
#endif
#if SPEAD2_USE_AVX_STREAM
    if (__builtin_cpu_supports("avx"))
        return memcpy_nontemporal_avx;
#endif
#if SPEAD2_USE_SSE2_STREAM
    if (__builtin_cpu_supports("sse2"))
        return memcpy_nontemporal_sse2;
#endif
    /* Depending on the C library, std::memcpy might or might not be marked
     * as noexcept. If not, we need this explicit cast.
     */
    return (void *(*)(void *, const void *, std::size_t) noexcept) std::memcpy;
}

#if SPEAD2_USE_FMV

[[gnu::ifunc("_ZN6spead226resolve_memcpy_nontemporalEv")]]
void *memcpy_nontemporal(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept;

#else

void *memcpy_nontemporal(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    static void *(*memcpy_nontemporal_ptr)(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept = resolve_memcpy_nontemporal();
    return memcpy_nontemporal_ptr(dest, src, n);
}

#endif

} // namespace spead2
