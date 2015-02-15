/**
 * @file
 *
 * Miscellaneous utilities for encoding SPEAD data.
 */

#ifndef SPEAD_SEND_UTILS_H
#define SPEAD_SEND_UTILS_H

#include <cstdint>
#include <cassert>

namespace spead
{
namespace send
{

class pointer_encoder
{
private:
    int heap_address_bits;

public:
    explicit pointer_encoder(int heap_address_bits)
        : heap_address_bits(heap_address_bits)
    {
        assert(heap_address_bits > 0);
        assert(heap_address_bits < 64);
        assert(heap_address_bits % 8 == 0);
    }

    std::uint64_t encode_immediate(std::int64_t id, std::int64_t value) const
    {
        assert(id >= 0 && id < (std::int64_t(1) << (63 - heap_address_bits)));
        assert(value >= 0 && value < (std::int64_t(1) << heap_address_bits));
        return (std::uint64_t(1) << 63) | (id << heap_address_bits) | value;
    }

    std::uint64_t encode_address(std::int64_t id, std::int64_t address) const
    {
        assert(id >= 0 && id < (std::int64_t(1) << (63 - heap_address_bits)));
        assert(address >= 0 && address < (std::int64_t(1) << heap_address_bits));
        return (id << heap_address_bits) | address;
    }
};

} // namespace send
} // namespace spead

#endif // SPEAD_SEND_UTILS_H
