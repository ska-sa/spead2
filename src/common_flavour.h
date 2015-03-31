/**
 * @file
 */

#ifndef SPEAD2_COMMON_FLAVOUR_H
#define SPEAD2_COMMON_FLAVOUR_H

#include "common_defines.h"

namespace spead2
{

/**
 * A variant of the SPEAD protocol.
 */
class flavour
{
private:
    int heap_address_bits = 40;
    bug_compat_mask bug_compat = 0;

public:
    flavour() = default;
    explicit flavour(
        int version, int item_pointer_bits,
        int heap_address_bits, bug_compat_mask bug_compat = 0);

    int get_version() const { return 4; }
    int get_item_pointer_bits() const { return 8 * sizeof(item_pointer_t); }
    int get_heap_address_bits() const { return heap_address_bits; }
    bug_compat_mask get_bug_compat() const { return bug_compat; }
};

} // namespace spead2

#endif // SPEAD2_COMMON_FLAVOUR_H
