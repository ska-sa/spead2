/* Copyright 2016 SKA South Africa
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

#ifndef SPEAD2_RECV_UDP_BASE_H
#define SPEAD2_RECV_UDP_BASE_H

#include <cstddef>
#include <cstdint>
#include <spead2/recv_reader.h>
#include <spead2/recv_stream.h>

namespace spead2
{
namespace recv
{

/**
 * Base class that has common logic between @ref udp_reader and @ref
 * udp_ibv_reader.
 */
class udp_reader_base : public reader
{
protected:
    /**
     * Handle a single received packet.
     *
     * @param state     Batch state
     * @param data      Pointer to the start of the UDP payload
     * @param length    Length of the UDP payload
     * @param max_size  Maximum expected length of the UDP payload
     *
     * @return whether the packet caused the stream to stop
     */
    bool process_one_packet(
        stream_base::add_packet_state &state,
        const std::uint8_t *data, std::size_t length, std::size_t max_size);

public:
    /// Maximum packet size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_max_size = 9200;

    using reader::reader;
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_UDP_BASE_H
