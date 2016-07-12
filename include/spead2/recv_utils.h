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
 *
 * Miscellaneous utilities for receiving SPEAD data.
 */

#ifndef SPEAD2_RECV_UTILS_H
#define SPEAD2_RECV_UTILS_H

#include <spead2/common_defines.h>

namespace spead2
{

/**
 * SPEAD stream receiver functionality.
 */
namespace recv
{

/**
 * Decodes an %ItemPointer into the ID, mode flag, and address/value.
 *
 * An %ItemPointer is encoded, from MSB to LSB, as
 * - a one bit mode flag (1 for immediate, 0 for address)
 * - an unsigned identifier
 * - either an integer value (in immediate mode) or a payload-relative address
 *   (in address mode).
 * The number of bits in the last field is given by @a heap_address_bits.
 *
 * The wire protocol uses big-endian, but this class assumes that the
 * conversion to host endian has already occurred.
 */
class pointer_decoder
{
private:
    int heap_address_bits;        ///< Bits for immediate/address field
    item_pointer_t address_mask;  ///< Mask selecting the immediate/address field
    item_pointer_t id_mask;       ///< Mask with number of bits for the ID field, shifted down

public:
    explicit pointer_decoder(int heap_address_bits)
    {
        this->heap_address_bits = heap_address_bits;
        this->address_mask = (item_pointer_t(1) << heap_address_bits) - 1;
        this->id_mask = (item_pointer_t(1) << (8 * sizeof(item_pointer_t) - 1 - heap_address_bits)) - 1;
    }

    /// Extract the ID from an item pointer
    s_item_pointer_t get_id(item_pointer_t pointer) const
    {
        return (pointer >> heap_address_bits) & id_mask;
    }

    /**
     * Extract the address from an item pointer. At present, no check is
     * done to ensure that the mode is correct.
     */
    s_item_pointer_t get_address(item_pointer_t pointer) const
    {
        return pointer & address_mask;
    }

    /**
     * Extract the immediate value from an item pointer. At present, no check
     * is done to ensure that the mode is correct.
     */
    s_item_pointer_t get_immediate(item_pointer_t pointer) const
    {
        return get_address(pointer);
    }

    /// Determine whether the item pointer uses immediate mode
    bool is_immediate(item_pointer_t pointer) const
    {
        return pointer >> (8 * sizeof(item_pointer_t) - 1);
    }

    /// Return the number of bits for address/immediate given to the constructor
    int address_bits() const
    {
        return heap_address_bits;
    }
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_UTILS_H
