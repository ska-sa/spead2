/* Copyright 2015 SKA South Africa
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
 */

#ifndef SPEAD2_COMMON_ENDIAN_H
#define SPEAD2_COMMON_ENDIAN_H

#include <cstdint>
#include <cstring>
#include <spead2/portable_endian.h>

namespace spead2
{

namespace detail
{

template<typename T>
struct Endian
{
};

template<>
struct Endian<std::uint64_t>
{
    static std::uint64_t htobe(std::uint64_t in)
    {
        return htobe64(in);
    }

    static std::uint64_t betoh(std::uint64_t in)
    {
        return be64toh(in);
    }
};

template<>
struct Endian<std::uint32_t>
{
    static std::uint32_t htobe(std::uint32_t in)
    {
        return htobe32(in);
    }

    static std::uint32_t betoh(std::uint32_t in)
    {
        return be32toh(in);
    }
};

template<>
struct Endian<std::uint16_t>
{
    static std::uint16_t htobe(std::uint16_t in)
    {
        return htobe16(in);
    }

    static std::uint16_t betoh(std::uint16_t in)
    {
        return be16toh(in);
    }
};

} // namespace detail

template<typename T>
static inline T htobe(T in)
{
    return detail::Endian<T>::htobe(in);
}

template<typename T>
static inline T betoh(T in)
{
    return detail::Endian<T>::betoh(in);
}

/**
 * Load a big-endian value stored at address @a ptr (not necessarily aligned).
 */
template<typename T>
static inline T load_be(const uint8_t *ptr)
{
    T out;
    std::memcpy(&out, ptr, sizeof(T));
    return betoh(out);
}

} // namespace spead2

#endif // SPEAD2_COMMON_ENDIAN_H
