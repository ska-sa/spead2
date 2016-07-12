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

#ifndef SPEAD2_COMMON_FLAVOUR_H
#define SPEAD2_COMMON_FLAVOUR_H

#include <spead2/common_defines.h>

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

    bool operator==(const flavour &other) const;
    bool operator!=(const flavour &other) const;
};

} // namespace spead2

#endif // SPEAD2_COMMON_FLAVOUR_H
