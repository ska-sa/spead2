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

#include <cstdint>
#include <functional>
#include <boost/asio.hpp>
#include <iostream>
#include "recv_reader.h"
#include "recv_udp.h"
#include "common_logging.h"

namespace spead2
{
namespace recv
{

constexpr std::size_t udp_reader::default_max_size;
constexpr std::size_t udp_reader::default_buffer_size;

udp_reader::udp_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size)
    : reader(owner), socket(get_io_service()), max_size(max_size)
{
    // Allocate one extra byte so that overflow can be detected
    buffer.reset(new std::uint8_t[max_size + 1]);
    socket.open(endpoint.protocol());
    if (buffer_size != 0)
    {
        boost::asio::socket_base::receive_buffer_size option(buffer_size);
        boost::system::error_code ec;
        socket.set_option(option, ec);
        if (ec)
        {
            log_warning("request for buffer size %s failed (%s): refer to documentation for details on increasing buffer size",
                        buffer_size, ec.message());
        }
        else
        {
            // Linux silently clips to the maximum allowed size
            boost::asio::socket_base::receive_buffer_size actual;
            socket.get_option(actual);
            if (std::size_t(actual.value()) < buffer_size)
            {
                log_warning("requested buffer size %d but only received %d: refer to documentation for details on increasing buffer size",
                            buffer_size, actual.value());
            }
        }
    }
    socket.bind(endpoint);
    enqueue_receive();
}

void udp_reader::packet_handler(
    const boost::system::error_code &error,
    std::size_t bytes_transferred)
{
    if (!error)
    {
        if (get_stream_base().is_stopped())
        {
            log_debug("UDP reader: discarding packet received after stream stopped");
        }
        else if (bytes_transferred <= max_size && bytes_transferred > 0)
        {
            // If it's bigger, the packet might have been truncated
            packet_header packet;
            std::size_t size = decode_packet(packet, buffer.get(), bytes_transferred);
            if (size == bytes_transferred)
            {
                get_stream_base().add_packet(packet);
                if (get_stream_base().is_stopped())
                {
                    log_debug("UDP reader: end of stream detected");
                }
            }
        }
        else if (bytes_transferred > max_size)
            log_debug("dropped packet due to truncation");
    }
    // TODO: log the error if there was one

    if (!get_stream_base().is_stopped())
    {
        enqueue_receive();
    }
    else
        stopped();
}

void udp_reader::enqueue_receive()
{
    using namespace std::placeholders;
    socket.async_receive_from(
        boost::asio::buffer(buffer.get(), max_size + 1),
        endpoint,
        get_stream().get_strand().wrap(std::bind(&udp_reader::packet_handler, this, _1, _2)));
}

void udp_reader::stop()
{
    /* asio guarantees that closing a socket will cancel any pending
     * operations on it.
     * Don't put any logging here: it could be running in a shutdown
     * path where it is no longer safe to do so.
     */
    socket.close();
}

} // namespace recv
} // namespace spead2
