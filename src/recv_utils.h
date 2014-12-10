#ifndef SPEAD_RECV_UTILS_H
#define SPEAD_RECV_UTILS_H

namespace spead
{
namespace recv
{

class pointer_decoder
{
private:
    int heap_address_bits;
    std::uint64_t address_mask;
    std::uint64_t value_mask;

public:
    explicit pointer_decoder(int heap_address_bits)
    {
        this->heap_address_bits = heap_address_bits;
        this->address_mask = (std::uint64_t(1) << heap_address_bits) - 1;
        this->value_mask = (std::uint64_t(1) << (63 - heap_address_bits)) - 1;
    }

    std::int64_t get_id(std::uint64_t pointer) const
    {
        return (pointer >> heap_address_bits) & value_mask;
    }

    std::int64_t get_address(std::uint64_t pointer) const
    {
        return pointer & address_mask;
    }

    std::int64_t get_value(std::uint64_t pointer) const
    {
        return get_address(pointer);
    }

    bool is_immediate(std::uint64_t pointer) const
    {
        return pointer >> 63;
    }

    int address_bits() const
    {
        return heap_address_bits;
    }
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_UTILS_H
