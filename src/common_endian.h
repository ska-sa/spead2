/**
 * @file
 */

#ifndef SPEAD2_COMMON_ENDIAN_H
#define SPEAD2_COMMON_ENDIAN_H

#include <endian.h>
#include <cstdint>

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

} // namespace spead2

#endif // SPEAD2_COMMON_ENDIAN_H
