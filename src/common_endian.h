/**
 * @file
 */

#ifndef SPEAD_COMMON_ENDIAN_H
#define SPEAD_COMMON_ENDIAN_H

#include <endian.h>
#include <cstdint>

namespace spead
{

template<typename T>
static inline T htobe(T in) = delete;

template<typename T>
static inline T betoh(T in) = delete;

template<>
inline std::uint64_t htobe<std::uint64_t>(std::uint64_t in)
{
    return ::htobe64(in);
}

template<>
inline std::uint64_t betoh<std::uint64_t>(std::uint64_t in)
{
    return ::be64toh(in);
}

} // namespace spead

#endif // SPEAD_COMMON_ENDIAN_H
