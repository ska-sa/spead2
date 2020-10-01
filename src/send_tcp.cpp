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

namespace detail
{

boost::asio::ip::tcp::socket make_socket(
    const io_service_ref &io_service,
    const std::vector<boost::asio::ip::tcp::endpoint> &endpoints,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
{
    if (endpoints.size() != 1)
        throw std::invalid_argument("endpoints must contain exactly one element");
    const boost::asio::ip::tcp::endpoint &endpoint = endpoints[0];
    boost::asio::ip::tcp::socket socket(*io_service, endpoint.protocol());
    if (!interface_address.is_unspecified())
        socket.bind(boost::asio::ip::tcp::endpoint(interface_address, 0));
    set_socket_send_buffer_size(socket, buffer_size);
    return socket;
}

} // namespace detail


void tcp_writer::wakeup()
{
    transmit_packet data;
    packet_result result = get_packet(data);
    switch (result)
    {
    case packet_result::SLEEP:
        sleep();
        return;
    case packet_result::EMPTY:
        request_wakeup();
        return;
    case packet_result::SUCCESS:
        break;
    }

    stream2::queue_item *item = data.item;
    bool last = data.last;
    auto handler = [this, item, last](const boost::system::error_code &ec, std::size_t bytes_transferred)
    {
        item->bytes_sent += bytes_transferred;
        if (!item->result)
            item->result = ec;
        if (last)
            heaps_completed(1);
        wakeup();
    };
    socket.async_send(data.pkt.buffers, std::move(handler));
}

void tcp_writer::start()
{
    if (!pre_connected)
    {
        socket.async_connect(endpoint,
            [this] (const boost::system::error_code &ec)
            {
                connect_handler(ec);
                wakeup();
            });
    }
    else
        request_wakeup();
}

tcp_writer::tcp_writer(
    io_service_ref io_service,
    std::function<void(const boost::system::error_code &)> &&connect_handler,
    const std::vector<boost::asio::ip::tcp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
    : writer(std::move(io_service), config),
    socket(detail::make_socket(get_io_service(), endpoints, buffer_size, interface_address)),
    pre_connected(false),
    endpoint(endpoints[0]),
    connect_handler(std::move(connect_handler))
{
}

tcp_writer::tcp_writer(
    io_service_ref io_service,
    boost::asio::ip::tcp::socket &&socket,
    const stream_config &config)
    : writer(std::move(io_service), config),
    socket(std::move(socket)),
    pre_connected(true)
{
    if (!socket_uses_io_service(this->socket, get_io_service()))
        throw std::invalid_argument("I/O service does not match the socket's I/O service");
}

constexpr std::size_t tcp_stream::default_buffer_size;

tcp_stream::tcp_stream(
    io_service_ref io_service,
    std::function<void(const boost::system::error_code &)> &&connect_handler,
    const std::vector<boost::asio::ip::tcp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
    : stream2(std::unique_ptr<writer>(new tcp_writer(
        std::move(io_service),
        std::move(connect_handler),
        endpoints,
        config,
        buffer_size,
        interface_address)))
{
}

tcp_stream::tcp_stream(
    io_service_ref io_service,
    boost::asio::ip::tcp::socket &&socket,
    const stream_config &config)
    : stream2(std::unique_ptr<writer>(new tcp_writer(
        std::move(io_service),
        std::move(socket),
        config)))
{
}

} // namespace send
} // namespace spead2
