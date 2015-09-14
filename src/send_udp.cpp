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
#include "send_udp.h"
#include "common_defines.h"

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
    : stream<udp_stream>(io_service, config),
    socket(io_service), endpoint(endpoint)
{
    boost::asio::ip::udp::socket socket(io_service);
    socket.open(endpoint.protocol());
    if (buffer_size != 0)
    {
        boost::asio::socket_base::send_buffer_size option(buffer_size);
        boost::system::error_code ec;
        socket.set_option(option, ec);
        if (ec)
        {
            log_warning("request for socket buffer size %s failed (%s): refer to documentation for details on increasing buffer size",
                        buffer_size, ec.message());
        }
        else
        {
            // Linux silently clips to the maximum allowed size
            boost::asio::socket_base::send_buffer_size actual;
            socket.get_option(actual);
            if (std::size_t(actual.value()) < buffer_size)
            {
                log_warning("requested socket buffer size %d but only received %d: refer to documentation for details on increasing buffer size",
                            buffer_size, actual.value());
            }
        }
    }
    this->socket = std::move(socket);
}

} // namespace send
} // namespace spead2
