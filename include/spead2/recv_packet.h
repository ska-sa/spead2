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

#ifndef SPEAD2_RECV_PACKET
#define SPEAD2_RECV_PACKET

#include <cstddef>
#include <cstdint>
#include <spead2/common_defines.h>

namespace spead2
{
namespace recv
{

/**
 * Unpacked packet header, with pointers to the original data.
 */
struct packet_header
{
    /// Number of bits in addresses/immediates (from SPEAD flavour)
    int heap_address_bits;
    /// Number of item pointers in the packet
    int n_items;
    /**
     * @name Key fields extracted from items in the packet
     * @{
     * The true values are always non-negative, and -1 is used to indicate
     * that the packet did not contain the item.
     */
    s_item_pointer_t heap_cnt;
    s_item_pointer_t heap_length;
    s_item_pointer_t payload_offset;
    s_item_pointer_t payload_length;
    /** @} */
    /// The item pointers in the packet, in big endian, and not necessarily aligned
    const std::uint8_t *pointers;
    /// Start of the packet payload
    const std::uint8_t *payload;
};

/**
 * Reads the size of the packet.
 *
 * @param  data   Start of packet
 * @param  length Size of data pointed to by @a data
 * @returns Actual packet size on success, 0 when size cannot be determined due
 * to truncation, and -1 on error due to malformed packet header.
 */
s_item_pointer_t get_packet_size(const uint8_t *data, std::size_t length);

/**
 * Split out the header fields for the packet.
 *
 * @param[out] out     Packet header with pointers to data (undefined on failure)
 * @param[in]  raw     Start of packet
 * @param      max_size Size of data pointed to by @a raw
 * @returns Actual packet size on success, or 0 on failure (due to malformed or
 * truncated packet).
 */
std::size_t decode_packet(packet_header &out, const std::uint8_t *raw, std::size_t max_size);

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_PACKET
