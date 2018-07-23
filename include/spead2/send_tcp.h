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
#include <stdexcept>
#include <utility>
#include <spead2/send_packet.h>
#include <spead2/send_stream.h>
#include <spead2/common_endian.h>
#include <spead2/common_socket.h>

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

    template<typename Handler>
    void async_send_packet(const packet &pkt, Handler &&handler)
    {
        boost::asio::async_write(socket, pkt.buffers, std::move(handler));
    }

public:
    /// Socket send buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 512 * 1024;

    /// Constructor
    template<typename ConnectHandler>
    tcp_stream(
        io_service_ref io_service,
        ConnectHandler &&connect_handler,
        const boost::asio::ip::tcp::endpoint &endpoint,
        const boost::asio::ip::tcp::endpoint &local_endpoint = boost::asio::ip::tcp::endpoint(),
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size)
        : tcp_stream(std::move(io_service),
                     boost::asio::ip::tcp::socket(*io_service, endpoint.protocol()),
                     std::forward<ConnectHandler>(connect_handler),
                     endpoint, local_endpoint, config, buffer_size)
    {
    }

    /**
     * Constructor using an existing socket. The socket must be open but
     * not bound.
     */
    template<typename ConnectHandler>
    tcp_stream(
        boost::asio::ip::tcp::socket &&socket,
        ConnectHandler &&connect_handler,
        const boost::asio::ip::tcp::endpoint &endpoint,
        const boost::asio::ip::tcp::endpoint &local_endpoint = boost::asio::ip::tcp::endpoint(),
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size)
        : tcp_stream(socket.get_io_service(), std::move(socket),
                     std::forward<ConnectHandler>(connect_handler),
                     endpoint, local_endpoint, config, buffer_size)
    {
    }


    /**
     * Constructor using an existing socket and an explicit io_service or
     * thread pool. The socket must be open but not bound, and the io_service
     * must match the socket's.
     */
    template<typename ConnectHandler>
    tcp_stream(
        io_service_ref io_service,
        boost::asio::ip::tcp::socket &&socket,
        ConnectHandler &&connect_handler,
        const boost::asio::ip::tcp::endpoint &endpoint,
        const boost::asio::ip::tcp::endpoint &local_endpoint = boost::asio::ip::tcp::endpoint(),
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size)
        : stream_impl<tcp_stream>(std::move(io_service), config),
          socket(std::move(socket)), endpoint(endpoint)
    {
        if (&get_io_service() != &this->socket.get_io_service())
            throw std::invalid_argument("I/O service does not match the socket's I/O service");
        set_socket_send_buffer_size(this->socket, buffer_size);
        if (!socket.is_open())
        {
            if (!local_endpoint.address().is_unspecified())
                this->socket.bind(local_endpoint);
            this->socket.async_connect(endpoint, connect_handler);
        }
    }


};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_TCP_H
