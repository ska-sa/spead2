/*
 * TCP sender for SPEAD protocol
 *
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia, 2018
 * Copyright by UWA (in the framework of the ICRAR)
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

#ifndef SPEAD2_SEND_TCP_H
#define SPEAD2_SEND_TCP_H

#include <boost/asio.hpp>
#include <utility>
#include <spead2/send_packet.h>
#include <spead2/send_stream.h>
#include <spead2/common_endian.h>

namespace spead2
{
namespace send
{

class tcp_stream : public stream_impl<tcp_stream>
{
private:
    friend class stream_impl<tcp_stream>;
    boost::asio::ip::tcp::socket socket;
    boost::asio::ip::tcp::endpoint endpoint;
    std::uint64_t packet_size;
    boost::asio::const_buffers_1 size_buffer;

    template<typename Handler>
    void async_send_packet(const packet &pkt, Handler &&handler)
    {
        packet_size = betoh(boost::asio::buffer_size(pkt.buffers));
        boost::asio::async_write(socket, size_buffer, [this, &pkt, handler] (const boost::system::error_code &ec, std::size_t bytes_transferred)
        {
            if (ec) {
                // not sure what to do really...
                return;
            }
            boost::asio::async_write(socket, pkt.buffers, std::move(handler));
        });
    }

public:
    /// Socket receive buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 512 * 1024;

    /// Constructor
    tcp_stream(
        io_service_ref io_service,
        const boost::asio::ip::tcp::endpoint &endpoint,
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size);

    /**
     * Constructor using an existing socket. The socket must be open but
     * not bound.
     */
    tcp_stream(
        boost::asio::ip::tcp::socket &&socket,
        const boost::asio::ip::tcp::endpoint &endpoint,
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size);

    /**
     * Constructor using an existing socket and an explicit io_service or
     * thread pool. The socket must be open but not bound, and the io_service
     * must match the socket's.
     */
    tcp_stream(
        io_service_ref io_service,
        boost::asio::ip::tcp::socket &&socket,
        const boost::asio::ip::tcp::endpoint &endpoint,
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size);

};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_TCP_H
