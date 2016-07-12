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

#include <stdexcept>
#include <spead2/common_flavour.h>
#include <spead2/common_defines.h>

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

bool flavour::operator==(const flavour &other) const
{
    return heap_address_bits == other.heap_address_bits
        && bug_compat == other.bug_compat;
}

bool flavour::operator!=(const flavour &other) const
{
    return !(*this == other);
}

} // namespace spead2
