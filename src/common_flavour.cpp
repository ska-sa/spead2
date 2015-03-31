/**
 * @file
 */

#include <stdexcept>
#include "common_flavour.h"
#include "common_defines.h"

namespace spead2
{

flavour::flavour(
    int version, int item_pointer_bits,
    int heap_address_bits, bug_compat_mask bug_compat)
{
    if (version != 4)
        throw std::invalid_argument("Version is not supported");
    if (item_pointer_bits != 8 * sizeof(item_pointer_t))
        throw std::invalid_argument("item_pointer_bits not supported");
    if (heap_address_bits <= 0 || heap_address_bits >= item_pointer_bits)
        throw std::invalid_argument("heap_address_bits out of range");
    if (heap_address_bits % 8 != 0)
        throw std::invalid_argument("heap_address_bits not a multiple of 8");

    this->heap_address_bits = heap_address_bits;
    this->bug_compat = bug_compat;
}

} // namespace spead2
