/* Copyright 2015, 2019-2020 SKA South Africa
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
#include <cstring>
#include <utility>
#include <boost/asio.hpp>
#include <spead2/send_udp.h>
#include <spead2/common_defines.h>
#include <spead2/common_socket.h>

namespace spead2
{
namespace send
{

constexpr std::size_t udp_stream::default_buffer_size;

static boost::asio::ip::udp get_protocol(const std::vector<boost::asio::ip::udp::endpoint> &endpoints)
{
    if (endpoints.empty())
        throw std::invalid_argument("Endpoint list must be non-empty");
    return endpoints[0].protocol();
}

void udp_stream::send_packets(std::size_t first)
{
#if SPEAD2_USE_SENDMMSG
    // Try synchronous send
    if (first < n_current_packets)
    {
        int sent = sendmmsg(socket.native_handle(), msgvec + first, n_current_packets - first, MSG_DONTWAIT);
        if (sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
        {
            current_packets[first].result = boost::system::error_code(errno, boost::asio::error::get_system_category());
            first++;
        }
        else if (sent > 0)
        {
            for (int i = 0; i < sent; i++)
                current_packets[first + i].result = boost::system::error_code();
            first += sent;
        }
        if (first < n_current_packets)
        {
            socket.async_send(boost::asio::null_buffers(), [this, first](const boost::system::error_code &ec, std::size_t)
            {
                send_packets(first);
            });
            return;
        }
    }
#else
    for (std::size_t idx = first; idx < n_current_packets; idx++)
    {
        // First try to send synchronously, to reduce overheads from callbacks etc
        boost::system::error_code ec;
        transmit_item &current = current_packets[idx];
        boost::asio::ip::udp::endpoint endpoint = endpoints[current.item->substream_index];
        socket.send_to(current.pkt.buffers, endpoint, 0, ec);
        if (ec == boost::asio::error::would_block)
        {
            // Socket buffer is full, fall back to asynchronous
            auto handler = [this, idx](const boost::system::error_code &ec, std::size_t bytes_transferred)
            {
                current_packets[idx].result = ec;
                send_packets(idx + 1);
            };
            socket.async_send_to(current.pkt.buffers, endpoint, handler);
            return;
        }
        else
        {
            current.result = ec;
        }
    }
#endif

    get_io_service().post([this] { packets_handler(); });
}

void udp_stream::async_send_packets()
{
#if SPEAD2_USE_SENDMMSG
    msg_iov.clear();
    for (std::size_t i = 0; i < n_current_packets; i++)
        for (const auto &buffer : current_packets[i].pkt.buffers)
        {
            msg_iov.push_back(iovec{const_cast<void *>(boost::asio::buffer_cast<const void *>(buffer)),
                                    boost::asio::buffer_size(buffer)});
        }
    // Assigning msgvec must be done in a second pass, because appending to
    // msg_iov invalidates references.
    std::size_t offset = 0;
    for (std::size_t i = 0; i < n_current_packets; i++)
    {
        auto &hdr = msgvec[i].msg_hdr;
        hdr.msg_iov = &msg_iov[offset];
        hdr.msg_iovlen = current_packets[i].pkt.buffers.size();
        offset += hdr.msg_iovlen;

        const auto &endpoint = endpoints[current_packets[i].item->substream_index];
        hdr.msg_name = (void *) endpoint.data();
        hdr.msg_namelen = endpoint.size();
    }
#endif
    send_packets(0);
}

static boost::asio::ip::udp::socket make_socket(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp &protocol,
    const boost::asio::ip::address &interface_address)
{
    boost::asio::ip::udp::socket socket(io_service, protocol);
    if (!interface_address.is_unspecified())
        socket.bind(boost::asio::ip::udp::endpoint(interface_address, 0));
    return socket;
}

udp_stream::udp_stream(
    io_service_ref io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
    : udp_stream(std::move(io_service),
                 make_socket(*io_service, get_protocol(endpoints), interface_address),
                 endpoints, config, buffer_size)
{
}

static boost::asio::ip::udp::socket make_multicast_socket(
    boost::asio::io_service &io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    int ttl)
{
    for (const auto &endpoint : endpoints)
        if (!endpoint.address().is_multicast())
            throw std::invalid_argument("endpoint is not a multicast address");
    boost::asio::ip::udp::socket socket(io_service, get_protocol(endpoints));
    socket.set_option(boost::asio::ip::multicast::hops(ttl));
    return socket;
}

static boost::asio::ip::udp::socket make_multicast_v4_socket(
    boost::asio::io_service &io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    int ttl,
    const boost::asio::ip::address &interface_address)
{
    for (const auto &endpoint : endpoints)
        if (!endpoint.address().is_v4() || !endpoint.address().is_multicast())
            throw std::invalid_argument("endpoint is not an IPv4 multicast address");
    if (!interface_address.is_unspecified() && !interface_address.is_v4())
        throw std::invalid_argument("interface address is not an IPv4 address");
    boost::asio::ip::udp::socket socket(io_service, boost::asio::ip::udp::v4());
    socket.set_option(boost::asio::ip::multicast::hops(ttl));
    if (!interface_address.is_unspecified())
        socket.set_option(boost::asio::ip::multicast::outbound_interface(interface_address.to_v4()));
    return socket;
}

static boost::asio::ip::udp::socket make_multicast_v6_socket(
    boost::asio::io_service &io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    int ttl, unsigned int interface_index)
{
    for (const auto &endpoint : endpoints)
        if (!endpoint.address().is_v6() || !endpoint.address().is_multicast())
            throw std::invalid_argument("endpoint is not an IPv4 multicast address");
    boost::asio::ip::udp::socket socket(io_service, boost::asio::ip::udp::v6());
    socket.set_option(boost::asio::ip::multicast::hops(ttl));
    socket.set_option(boost::asio::ip::multicast::outbound_interface(interface_index));
    return socket;
}

udp_stream::udp_stream(
    io_service_ref io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    int ttl)
    : udp_stream(std::move(io_service),
                 make_multicast_socket(*io_service, endpoints, ttl),
                 std::move(endpoints), config, buffer_size)
{
}

udp_stream::udp_stream(
    io_service_ref io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    int ttl,
    const boost::asio::ip::address &interface_address)
    : udp_stream(std::move(io_service),
                 make_multicast_v4_socket(*io_service, endpoints, ttl, interface_address),
                 std::move(endpoints), config, buffer_size)
{
}

udp_stream::udp_stream(
    io_service_ref io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    int ttl,
    unsigned int interface_index)
    : udp_stream(std::move(io_service),
                 make_multicast_v6_socket(*io_service, endpoints, ttl, interface_index),
                 std::move(endpoints), config, buffer_size)
{
}

udp_stream::udp_stream(
    io_service_ref io_service,
    boost::asio::ip::udp::socket &&socket,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size)
    : stream_impl<udp_stream>(std::move(io_service), config, batch_size),
    socket(std::move(socket)), endpoints(endpoints)
{
    if (!socket_uses_io_service(this->socket, get_io_service()))
        throw std::invalid_argument("I/O service does not match the socket's I/O service");
    auto protocol = this->socket.local_endpoint().protocol();
    for (const auto &endpoint : endpoints)
        if (endpoint.protocol() != protocol)
            throw std::invalid_argument("Endpoint does not match protocol of the socket");
    set_socket_send_buffer_size(this->socket, buffer_size);
    this->socket.non_blocking(true);
#if SPEAD2_USE_SENDMMSG
    std::memset(&msgvec, 0, sizeof(msgvec));
#endif // SPEAD2_USE_SENDMMSG
}

udp_stream::udp_stream(
    io_service_ref io_service,
    boost::asio::ip::udp::socket &&socket,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config)
    : udp_stream(io_service, std::move(socket), endpoints, config, 0)
{
}

udp_stream::~udp_stream()
{
    flush();
}

} // namespace send
} // namespace spead2
