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
#include <utility>
#include <spead2/common_socket.h>
#include <spead2/send_tcp.h>
#include <spead2/send_writer.h>

namespace spead2::send
{

static boost::asio::ip::tcp::socket make_socket(
    const io_context_ref &io_context,
    const std::vector<boost::asio::ip::tcp::endpoint> &endpoints,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
{
    if (endpoints.size() != 1)
        throw std::invalid_argument("endpoints must contain exactly one element");
    const boost::asio::ip::tcp::endpoint &endpoint = endpoints[0];
    boost::asio::ip::tcp::socket socket(*io_context, endpoint.protocol());
    if (!interface_address.is_unspecified())
        socket.bind(boost::asio::ip::tcp::endpoint(interface_address, 0));
    set_socket_send_buffer_size(socket, buffer_size);
    return socket;
}

namespace
{

class tcp_writer : public writer
{
private:
    /// The underlying TCP socket
    boost::asio::ip::tcp::socket socket;
    /// Whether we were handled an already-connected socket
    const bool pre_connected;
    /// Endpoint to connect to (if not pre-connected)
    boost::asio::ip::tcp::endpoint endpoint;
    /// Callback once connected (if not pre-connected)
    std::function<void(const boost::system::error_code &)> connect_handler;
    // Scratch space for constructing packets
    std::unique_ptr<std::uint8_t[]> scratch;

    virtual void wakeup() override final;
    virtual void start() override final;

public:
    /**
     * Constructor. A callback is provided to indicate when the connection is
     * established.
     *
     * @warning The callback may be called before the constructor returns. The
     * implementation of the callback needs to be prepared to handle this case.
     *
     * @param io_context   I/O context for sending data
     * @param connect_handler  Callback when connection is established. It is called
     *                     with a @c boost::system::error_code to indicate whether
     *                     connection was successful.
     * @param endpoints    Destination host and port (must contain exactly one element)
     * @param config       Stream configuration
     * @param buffer_size  Socket buffer size (0 for OS default)
     * @param interface_address   Source address
     *                            @verbatim embed:rst:leading-asterisks
     *                            (see tips on :ref:`routing`)
     *                            @endverbatim
     */
    tcp_writer(
        io_context_ref io_context,
        std::function<void(const boost::system::error_code &)> &&connect_handler,
        const std::vector<boost::asio::ip::tcp::endpoint> &endpoints,
        const stream_config &config,
        std::size_t buffer_size,
        const boost::asio::ip::address &interface_address);

    /**
     * Constructor using an existing socket. The socket must be connected.
     */
    tcp_writer(
        io_context_ref io_context,
        boost::asio::ip::tcp::socket &&socket,
        const stream_config &config);

    virtual std::size_t get_num_substreams() const override final { return 1; }
};

void tcp_writer::wakeup()
{
    transmit_packet data;
    packet_result result = get_packet(data, scratch.get());
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

    auto *item = data.item;
    bool last = data.last;
    auto handler = [this, item, last](const boost::system::error_code &ec, std::size_t bytes_transferred)
    {
        item->bytes_sent += bytes_transferred;
        if (!item->result)
            item->result = ec;
        if (last)
            groups_completed(1);
        wakeup();
    };
    boost::asio::async_write(socket, data.buffers, std::move(handler));
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
    io_context_ref io_context,
    std::function<void(const boost::system::error_code &)> &&connect_handler,
    const std::vector<boost::asio::ip::tcp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
    : writer(std::move(io_context), config),
    socket(make_socket(get_io_context(), endpoints, buffer_size, interface_address)),
    pre_connected(false),
    endpoint(endpoints[0]),
    connect_handler(std::move(connect_handler)),
    scratch(new std::uint8_t[config.get_max_packet_size()])
{
}

tcp_writer::tcp_writer(
    io_context_ref io_context,
    boost::asio::ip::tcp::socket &&socket,
    const stream_config &config)
    : writer(std::move(io_context), config),
    socket(std::move(socket)),
    pre_connected(true),
    scratch(new std::uint8_t[config.get_max_packet_size()])
{
    if (!socket_uses_io_context(this->socket, get_io_context()))
        throw std::invalid_argument("I/O context does not match the socket's I/O context");
}

} // anonymous namespace

tcp_stream::tcp_stream(
    io_context_ref io_context,
    std::function<void(const boost::system::error_code &)> &&connect_handler,
    const std::vector<boost::asio::ip::tcp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
    : stream(std::make_unique<tcp_writer>(
        std::move(io_context),
        std::move(connect_handler),
        endpoints,
        config,
        buffer_size,
        interface_address))
{
}

tcp_stream::tcp_stream(
    io_context_ref io_context,
    boost::asio::ip::tcp::socket &&socket,
    const stream_config &config)
    : stream(std::make_unique<tcp_writer>(
        std::move(io_context),
        std::move(socket),
        config))
{
}

} // namespace spead2::send
