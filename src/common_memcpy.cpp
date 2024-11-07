/* Copyright 2016, 2020, 2023-2024 National Research Foundation (SARAO)
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
# include "common_memcpy_x86.h"
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
# include "common_memcpy_x86.h"
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
# include "common_memcpy_x86.h"
#endif

#if SPEAD2_USE_SVE_STREAM
# include <atomic>
# include <sys/auxv.h>
# include <arm_sve.h>
#endif

namespace spead2
{

#if SPEAD2_USE_SVE_STREAM
[[gnu::target("+sve")]]
void *memcpy_nontemporal_sve(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    /* The AArch64 memory model says
     *
     * "If an address dependency exists between two Read Memory and an SVE
     * non-temporal vector load instruction generated the second read, then in
     * the absence of any other barrier mechanism to achieve order, the memory
     * accesses can be observed in any order by the other observers within the
     * shareability domain of the memory addresses being accessed."
     *
     * I think that in the C++ memory model, this should only affect
     * std::memory_order_consume (since "carries dependency" is the only time
     * reads are assumed to be ordered in the absence of explicit
     * synchronisation); memory_order_consume is not used anywhere in spead2,
     * the C++ standard discourages it, and it's believed that no compiler
     * actually implements it other than by upgrade to acquire.
     *
     * The user documentation for @ref memcpy_nontemporal indicates this
     * limitation, so we do not insert any barriers here. If it becomes
     * necessary in future, testing on a Grace GH200 (Neoverse V2) chip
     * suggests that it is more efficient to write the address to an atomic
     * and read it back with memory_order_acquire than it is to use
     * atomic_thread_fence.
     */

    std::uint8_t *destc = (std::uint8_t *) dest;
    const std::uint8_t *srcc = (const std::uint8_t *) src;
    std::size_t i = 0;  // byte offset for next copy

    /* Alignment requires we have data up to the next multiple, and it's
     * not worth unrolling unless we have a reasonable amount of data.
     * For anything smaller, we'll just rely on the tail handling.
     */
    if (n >= 4 * svcntb())
    {
        /* Align the source pointer to a multiple of the vector size.
         * Experiments on Grace (Neoverse V2) show that source alignment
         * is more important than destination alignment to throughput.
         *
         * C++ doesn't guarantee the representation of a pointer when
         * cast to uintptr_t, but we're only depending on it for performance,
         * not correctness.
         */
        std::size_t head = -std::uintptr_t(src) & (svcntb() - 1);
        svbool_t pg = svwhilelt_b8(i, head);
        svstnt1_u8(pg, destc, svldnt1_u8(pg, srcc));
        i = head;

        while (i + 2 * svcntb() <= n)
        {
            svuint8_t data0 = svldnt1_u8(svptrue_b8(), &srcc[i]);
            svuint8_t data1 = svldnt1_u8(svptrue_b8(), &srcc[i + svcntb()]);
            svstnt1_u8(svptrue_b8(), &destc[i], data0);
            svstnt1_u8(svptrue_b8(), &destc[i + svcntb()], data1);
            i += 2 * svcntb();
        }
    }

    svbool_t pg = svwhilelt_b8(i, n);
    do
    {
        svstnt1_u8(pg, &destc[i], svldnt1_u8(pg, &srcc[i]));
        i += svcntb();
    } while (svptest_first(svptrue_b8(), pg = svwhilelt_b8(i, n)));
    return dest;
}
#endif // SPEAD2_USE_SVE_STREAM

extern "C" void *(*spead2_resolve_memcpy_nontemporal(
#if SPEAD2_USE_SVE_STREAM
    std::uint64_t hwcaps  // See System V AVI for AArch64
#endif
))(void *, const void *, std::size_t) noexcept
{
    /* x86 options */
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

    /* aarch64 options */
#if SPEAD2_USE_SVE_STREAM
    if (hwcaps & HWCAP_SVE)
        return memcpy_nontemporal_sve;
#endif

    /* Depending on the C library, std::memcpy might or might not be marked
     * as noexcept. If not, we need this explicit cast.
     */
    return (void *(*)(void *, const void *, std::size_t) noexcept) std::memcpy;
}

#if SPEAD2_USE_FMV

[[gnu::ifunc("spead2_resolve_memcpy_nontemporal")]]
void *memcpy_nontemporal(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept;

#else

void *memcpy_nontemporal(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
#if SPEAD2_USE_SVE_STREAM
    static void *(*memcpy_nontemporal_ptr)(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept =
        spead2_resolve_memcpy_nontemporal(getauxval(AT_HWCAP));
#else
    static void *(*memcpy_nontemporal_ptr)(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept =
        spead2_resolve_memcpy_nontemporal();
#endif
    return memcpy_nontemporal_ptr(dest, src, n);
}

#endif

} // namespace spead2
