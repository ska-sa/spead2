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
#include <spead2/common_socket.h>

namespace spead2
{
namespace send
{

namespace detail
{

void prepare_socket(
    boost::asio::ip::tcp::socket &socket,
    const boost::asio::ip::tcp::endpoint &local_endpoint,
    std::size_t buffer_size);

template<typename ConnectHandler>
boost::asio::ip::tcp::socket make_socket(
    const io_service_ref &io_service,
    const boost::asio::ip::tcp::endpoint &remote_endpoint,
    const boost::asio::ip::tcp::endpoint &local_endpoint,
    std::size_t buffer_size,
    ConnectHandler &&connect_handler)
{
    boost::asio::ip::tcp::socket socket(*io_service, remote_endpoint.protocol());
    prepare_socket(socket, local_endpoint, buffer_size);
    socket.async_connect(remote_endpoint, connect_handler);
    return socket;
}

} // namespace detail

class tcp_stream : public stream_impl<tcp_stream>
{
private:
    friend class stream_impl<tcp_stream>;

    /// The underlying TCP socket
    boost::asio::ip::tcp::socket socket;
    /// Whether the underlying socket is already connected or not
    std::atomic<bool> connected{false};

    /// Constructor taking a properly configured socket
    tcp_stream(
        io_service_ref io_service,
        boost::asio::ip::tcp::socket &&socket,
        const stream_config &config,
        bool already_connected);

    template<typename Handler>
    void async_send_packet(const packet &pkt, Handler &&handler)
    {
        if (!connected.load())
            handler(boost::asio::error::not_connected, 0);
        else
            boost::asio::async_write(socket, pkt.buffers, std::move(handler));
    }

public:
    /// Socket send buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 208 * 1024;

    /// Constructor
    template<typename ConnectHandler>
    tcp_stream(
        io_service_ref io_service,
        ConnectHandler &&connect_handler,
        const boost::asio::ip::tcp::endpoint &remote_endpoint,
        const boost::asio::ip::tcp::endpoint &local_endpoint = boost::asio::ip::tcp::endpoint(),
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size)
        : tcp_stream(
            io_service,
            detail::make_socket(io_service, remote_endpoint, local_endpoint, buffer_size,
                [this, connect_handler] (boost::system::error_code ec)
                {
                    if (!ec)
                        connected.store(true);
                    connect_handler(ec);
                }),
            config, false)
    {
    }

    /**
     * Constructor using an existing socket. The socket must be connected.
     */
    tcp_stream(
        boost::asio::ip::tcp::socket &&socket,
        const stream_config &config = stream_config());

    /**
     * Constructor using an existing socket. The socket must be connected.
     */
    tcp_stream(
        io_service_ref io_service,
        boost::asio::ip::tcp::socket &&socket,
        const stream_config &config = stream_config());
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_TCP_H
