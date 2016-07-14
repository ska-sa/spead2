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

#include <cstddef>
#include <utility>
#include <boost/asio.hpp>
#include <spead2/send_udp.h>
#include <spead2/common_defines.h>

namespace spead2
{
namespace send
{

constexpr std::size_t udp_stream::default_buffer_size;

udp_stream::udp_stream(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    const stream_config &config,
    std::size_t buffer_size)
    : udp_stream(boost::asio::ip::udp::socket(io_service, endpoint.protocol()),
                 endpoint, config, buffer_size)
{
}

static boost::asio::ip::udp::socket make_multicast_socket(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    int ttl)
{
    if (!endpoint.address().is_multicast())
        throw std::invalid_argument("endpoint is not a multicast address");
    boost::asio::ip::udp::socket socket(io_service, endpoint.protocol());
    socket.set_option(boost::asio::ip::multicast::hops(ttl));
    return socket;
}

static boost::asio::ip::udp::socket make_multicast_v4_socket(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    int ttl,
    const boost::asio::ip::address &interface_address)
{
    if (!endpoint.address().is_v4() || !endpoint.address().is_multicast())
        throw std::invalid_argument("endpoint is not an IPv4 multicast address");
    if (!interface_address.is_v4())
        throw std::invalid_argument("interface address is not an IPv4 address");
    boost::asio::ip::udp::socket socket(io_service, endpoint.protocol());
    socket.set_option(boost::asio::ip::multicast::hops(ttl));
    socket.set_option(boost::asio::ip::multicast::outbound_interface(interface_address.to_v4()));
    return socket;
}

static boost::asio::ip::udp::socket make_multicast_v6_socket(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    int ttl, unsigned int interface_index)
{
    if (!endpoint.address().is_v6() || !endpoint.address().is_multicast())
        throw std::invalid_argument("endpoint is not an IPv4 multicast address");
    boost::asio::ip::udp::socket socket(io_service, endpoint.protocol());
    socket.set_option(boost::asio::ip::multicast::hops(ttl));
    socket.set_option(boost::asio::ip::multicast::outbound_interface(interface_index));
    return socket;
}

udp_stream::udp_stream(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    const stream_config &config,
    std::size_t buffer_size,
    int ttl)
    : udp_stream(make_multicast_socket(io_service, endpoint, ttl),
                 endpoint, config, buffer_size)
{
}

udp_stream::udp_stream(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    const stream_config &config,
    std::size_t buffer_size,
    int ttl,
    const boost::asio::ip::address &interface_address)
    : udp_stream(make_multicast_v4_socket(io_service, endpoint, ttl, interface_address),
                 endpoint, config, buffer_size)
{
}

udp_stream::udp_stream(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    const stream_config &config,
    std::size_t buffer_size,
    int ttl,
    unsigned int interface_index)
    : udp_stream(make_multicast_v6_socket(io_service, endpoint, ttl, interface_index),
                 endpoint, config, buffer_size)
{
}

udp_stream::udp_stream(
    boost::asio::ip::udp::socket &&socket,
    const boost::asio::ip::udp::endpoint &endpoint,
    const stream_config &config,
    std::size_t buffer_size)
    : stream_impl<udp_stream>(socket.get_io_service(), config),
    socket(std::move(socket)), endpoint(endpoint)
{
    if (buffer_size != 0)
    {
        boost::asio::socket_base::send_buffer_size option(buffer_size);
        boost::system::error_code ec;
        this->socket.set_option(option, ec);
        if (ec)
        {
            log_warning("request for socket buffer size %s failed (%s): refer to documentation for details on increasing buffer size",
                        buffer_size, ec.message());
        }
        else
        {
            // Linux silently clips to the maximum allowed size
            boost::asio::socket_base::send_buffer_size actual;
            this->socket.get_option(actual);
            if (std::size_t(actual.value()) < buffer_size)
            {
                log_warning("requested socket buffer size %d but only received %d: refer to documentation for details on increasing buffer size",
                            buffer_size, actual.value());
            }
        }
    }
}

} // namespace send
} // namespace spead2
