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

/**
 * @file
 *
 * Non-temporal memcpy implementation. This header file is included
 * multiple times, with the including code providing different macros each
 * time. While C++ metaprogramming would have been preferable, GCC does not
 * provide a way to specialise a template function based on the target
 * options.
 */

namespace spead2
{

namespace detail
{

/* U1 and U2 are unrolling factors; Ix is an index sequence 0, ..., Ux-1
 *
 * The first unrolling factor (U1) is for the bulk of the copy. The second
 * is for the tail, and should correspond to a cache line.
 */
template<int U1, int U2, std::size_t... I1, std::size_t... I2 >
[[gnu::target(SPEAD2_MEMCPY_TARGET)]]
static void *SPEAD2_MEMCPY_NAME(
    void * __restrict__ dest, const void * __restrict__ src, std::size_t n,
    std::index_sequence<I1...>,
    std::index_sequence<I2...>) noexcept
{
    using T = SPEAD2_MEMCPY_TYPE;
    char * __restrict__ dest_c = (char *) dest;
    const char * __restrict__ src_c = (const char *) src;
    // Align the destination to a cache-line boundary
    std::uintptr_t dest_i = std::uintptr_t(dest_c);
    constexpr std::uintptr_t cache_line_mask = cache_line_size - 1;
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
    std::size_t offset = 0;
    for (; offset + U1 * sizeof(T) <= n; offset += U1 * sizeof(T))
    {
        T values[U1];
        /* These fold expressions are really just loops in disguise. They're used
         * because GCC at -O2 doesn't do a good job of unrolling the loop.
         */
        ((values[I1] = SPEAD2_MEMCPY_LOAD((const T *) (src_c + offset + I1 * sizeof(T)))), ...);
        (SPEAD2_MEMCPY_STORE((T *) (dest_c + offset + I1 * sizeof(T)), values[I1]), ...);
    }
    if constexpr (U2 < U1)
    {
        for (; offset + U2 * sizeof(T) <= n; offset += U2 * sizeof(T))
        {
            T values[U2];
            /* These fold expressions are really just loops in disguise. They're used
             * because GCC at -O2 doesn't do a good job of unrolling the loop.
             */
            ((values[I2] = SPEAD2_MEMCPY_LOAD((const T *) (src_c + offset + I2 * sizeof(T)))), ...);
            (SPEAD2_MEMCPY_STORE((T *) (dest_c + offset + I2 * sizeof(T)), values[I2]), ...);
        }
    }
#if SPEAD2_MEMCPY_VZEROUPPER
    _mm256_zeroupper();
#endif
    std::size_t tail = n - offset;
    std::memcpy(dest_c + offset, src_c + offset, tail);
    _mm_sfence();
    return dest;
}

} // namespace detail

[[gnu::target(SPEAD2_MEMCPY_TARGET)]]
void *SPEAD2_MEMCPY_NAME(void * __restrict__ dest, const void * __restrict__ src, std::size_t n) noexcept
{
    constexpr std::size_t unroll2 = detail::cache_line_size / sizeof(SPEAD2_MEMCPY_TYPE);
    return detail::SPEAD2_MEMCPY_NAME<SPEAD2_MEMCPY_UNROLL, unroll2>(
        dest, src, n,
        std::make_index_sequence<SPEAD2_MEMCPY_UNROLL>(),
        std::make_index_sequence<unroll2>()
    );
}

} // namespace spead2

#undef SPEAD2_MEMCPY_NAME
#undef SPEAD2_MEMCPY_TARGET
#undef SPEAD2_MEMCPY_TYPE
#undef SPEAD2_MEMCPY_LOAD
#undef SPEAD2_MEMCPY_STORE
#undef SPEAD2_MEMCPY_UNROLL
#undef SPEAD2_MEMCPY_VZEROUPPER
