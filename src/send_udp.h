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

#ifndef SPEAD2_SEND_UDP_H
#define SPEAD2_SEND_UDP_H

#include <boost/asio.hpp>
#include <utility>
#include "send_packet.h"
#include "send_stream.h"

namespace spead2
{
namespace send
{

class udp_stream : public stream<udp_stream>
{
private:
    friend class stream<udp_stream>;
    boost::asio::ip::udp::socket socket;
    boost::asio::ip::udp::endpoint endpoint;

    template<typename Handler>
    void async_send_packet(const packet &pkt, Handler &&handler)
    {
        socket.async_send_to(pkt.buffers, endpoint, std::move(handler));
    }

public:
    /// Socket receive buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 512 * 1024;

    /// Constructor
    udp_stream(
        boost::asio::io_service &io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size);
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_UDP_H
