/* Copyright 2015, 2019-2020 National Research Foundation (SARAO)
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

#ifndef SPEAD2_SEND_PACKET_H
#define SPEAD2_SEND_PACKET_H

#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <boost/asio/buffer.hpp>
#include <spead2/common_defines.h>

namespace spead2
{
namespace send
{

class heap;

class packet_generator
{
private:
    // 8 bytes header, item pointers for heap cnt, heap size, payload offset, payload size
    static constexpr std::size_t prefix_size = 8 + 4 * sizeof(item_pointer_t);

    const heap &h;
    const item_pointer_t cnt;
    const std::size_t max_packet_size;
    const std::size_t max_item_pointers_per_packet;

    /// @name Item pointer generation
    /// @{
    /// Next item pointer to send
    std::size_t next_item_pointer = 0;
    /// Address at which payload for the next item will be found
    std::size_t next_address = 0;
    /// @}

    /// @name Payload generation
    /// @{
    /// Current item payload being sent
    std::size_t next_item = 0;
    /// Amount of next_item already sent
    std::size_t next_item_offset = 0;
    /// Payload offset for the next packet
    s_item_pointer_t payload_offset = 0;
    s_item_pointer_t payload_size = 0;
    /// @}
    /// There is payload padding, so we need to add a NULL item pointer
    bool need_null_item = false;

public:
    packet_generator(const heap &h, item_pointer_t cnt, std::size_t max_packet_size);

    /**
     * The maximum size of a packet this generator will generate. It may be
     * smaller than the value passed to the constructor (for alignment
     * reasons), but will never be larger.
     */
    std::size_t get_max_packet_size() const;

    bool has_next_packet() const;
    /**
     * Create a packet ready for sending on the network. The caller must
     * provide space for storing data, of size at least
     * @ref get_max_packet_size, and with at least @c item_pointer_t alignment.
     * The first element of the returned buffer sequence will reference a
     * prefix of @a scratch, while the remaining elements will reference to the
     * items of the original heap.
     *
     * If there are no more packets to send for the heap, an empty list is
     * returned.
     */
    std::vector<boost::asio::const_buffer> next_packet(std::uint8_t *scratch);
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_PACKET_H
