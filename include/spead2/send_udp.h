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
#include <spead2/send_packet.h>
#include <spead2/send_stream.h>

namespace spead2
{
namespace send
{

class udp_stream : public stream_impl<udp_stream>
{
private:
    friend class stream_impl<udp_stream>;
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

    /**
     * Constructor using an existing socket. The socket must be open but
     * not bound.
     */
    udp_stream(
        boost::asio::ip::udp::socket &&socket,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size);

    /**
     * Constructor with multicast hop count.
     *
     * @param io_service   I/O service for sending data
     * @param endpoint     Multicast group and port
     * @param config       Stream configuration
     * @param buffer_size  Socket buffer size (0 for OS default)
     * @param ttl          Maximum number of hops
     *
     * @throws std::invalid_argument if @a endpoint is not a multicast address
     */
    udp_stream(
        boost::asio::io_service &io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl);

    /**
     * Constructor with multicast hop count and outgoing interface address
     * (IPv4 only).
     *
     * @param io_service   I/O service for sending data
     * @param endpoint     Multicast group and port
     * @param config       Stream configuration
     * @param buffer_size  Socket buffer size (0 for OS default)
     * @param ttl          Maximum number of hops
     * @param interface_address   Address of the outgoing interface
     *
     * @throws std::invalid_argument if @a endpoint is not an IPv4 multicast address
     * @throws std::invalid_argument if @a interface_address is not an IPv4 address
     */
    udp_stream(
        boost::asio::io_service &io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        const boost::asio::ip::address &interface_address);

    /**
     * Constructor with multicast hop count and outgoing interface address
     * (IPv6 only).
     *
     * @param io_service   I/O service for sending data
     * @param endpoint     Multicast group and port
     * @param config       Stream configuration
     * @param buffer_size  Socket buffer size (0 for OS default)
     * @param ttl          Maximum number of hops
     * @param interface_index   Index of the outgoing interface
     *
     * @throws std::invalid_argument if @a endpoint is not an IPv6 multicast address
     *
     * @see if_nametoindex(3)
     */
    udp_stream(
        boost::asio::io_service &io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        unsigned int interface_index);
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_UDP_H
