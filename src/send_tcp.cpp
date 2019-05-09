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

#include <stdexcept>
#include <spead2/send_tcp.h>

namespace spead2
{
namespace send
{

void tcp_stream::async_send_packets()
{
    if (!connected.load())
    {
        current_packets[0].result = boost::asio::error::not_connected;
        get_io_service().post([this] { packets_handler(); });
    }
    else
    {
        auto handler = [this](const boost::system::error_code &ec, std::size_t)
        {
            current_packets[0].result = ec;
            packets_handler();
        };
        boost::asio::async_write(socket, current_packets[0].pkt.buffers, handler);
    }
}

namespace detail
{

boost::asio::ip::tcp::socket make_socket(
    const io_service_ref &io_service,
    const boost::asio::ip::tcp::endpoint &endpoint,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
{
    boost::asio::ip::tcp::socket socket(*io_service, endpoint.protocol());
    if (!interface_address.is_unspecified())
        socket.bind(boost::asio::ip::tcp::endpoint(interface_address, 0));
    set_socket_send_buffer_size(socket, buffer_size);
    return socket;
}

} // namespace detail

constexpr std::size_t tcp_stream::default_buffer_size;

tcp_stream::tcp_stream(
    io_service_ref io_service,
    boost::asio::ip::tcp::socket &&socket,
    const stream_config &config)
    : stream_impl(std::move(io_service), config, 1),
    socket(std::move(socket)),
    connected(true)
{
    if (!socket_uses_io_service(this->socket, get_io_service()))
        throw std::invalid_argument("I/O service does not match the socket's I/O service");
}

#if BOOST_VERSION < 107000
tcp_stream::tcp_stream(
    boost::asio::ip::tcp::socket &&socket,
    const stream_config &config)
    : tcp_stream(get_socket_io_service(socket), std::move(socket), config)
{
}
#endif

tcp_stream::~tcp_stream()
{
    flush();
}

} // namespace send
} // namespace spead2
