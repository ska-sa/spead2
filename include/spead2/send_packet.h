/* Copyright 2015, 2019 SKA South Africa
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

/**
 * A packet ready for sending on the network. It contains some internally
 * held data, and const buffer sequence that contains a mix of pointers to
 * the internal data and pointers to the heap's items.
 *
 * If @a buffers is empty, it indicates the end of the heap.
 *
 * @todo Investigate whether number of new calls could be reduced by using
 * a pool for the case of packets with no item pointers other than the
 * per-packet ones.
 */
struct packet
{
    std::unique_ptr<std::uint8_t[]> data;
    std::vector<boost::asio::const_buffer> buffers;
};

class packet_generator
{
private:
    // 8 bytes header, item pointerh for heap cnt, heap size, payload offset, payload size
    static constexpr std::size_t prefix_size = 8 + 4 * sizeof(item_pointer_t);

    const heap &h;
    item_pointer_t cnt;
    std::size_t max_packet_size;
    std::size_t max_item_pointers_per_packet;

    /// Next item pointer to send
    std::size_t next_item_pointer = 0;
    /// Current item payload being sent
    std::size_t next_item = 0;
    /// Amount of next_item already sent
    std::size_t next_item_offset = 0;
    /// Address at which payload for the next item will be found
    std::size_t next_address = 0;
    /// Payload offset for the next packet
    s_item_pointer_t payload_offset = 0;
    s_item_pointer_t payload_size = 0;
    /// There is payload padding, so we need to add a NULL item pointer
    bool need_null_item = false;

public:
    packet_generator(const heap &h, item_pointer_t cnt, std::size_t max_packet_size);

    bool has_next_packet() const;
    packet next_packet();
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_PACKET_H
