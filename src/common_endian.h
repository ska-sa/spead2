/**
 * @file
 */

#ifndef SPEAD2_COMMON_ENDIAN_H
#define SPEAD2_COMMON_ENDIAN_H

#include <endian.h>
#include <cstdint>
#include <cstring>

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
