/*
 * Common socket operations
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

/**
 * @file
 */

#include <spead2/common_logging.h>
#include <spead2/common_socket.h>

namespace spead2
{

template<typename SocketType, typename BufferSizeOption>
static void set_socket_buffer_size(SocketType &socket, std::size_t buffer_size)
{
    if (buffer_size == 0)
        return;
    BufferSizeOption option(buffer_size);
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
        BufferSizeOption actual;
        socket.get_option(actual);
        if (std::size_t(actual.value()) < buffer_size)
        {
            log_warning("requested socket buffer size %d but only received %d: refer to documentation for details on increasing buffer size",
                        buffer_size, actual.value());
        }
    }
}

template<typename SocketType>
void set_socket_send_buffer_size(SocketType &socket, std::size_t buffer_size)
{
    set_socket_buffer_size<SocketType, boost::asio::socket_base::send_buffer_size>(socket, buffer_size);
}

template<typename SocketType>
void set_socket_recv_buffer_size(SocketType &socket, std::size_t buffer_size)
{
    set_socket_buffer_size<SocketType, boost::asio::socket_base::receive_buffer_size>(socket, buffer_size);
}

template void set_socket_send_buffer_size<boost::asio::ip::tcp::socket>(boost::asio::ip::tcp::socket &socket, std::size_t buffer_size);
template void set_socket_send_buffer_size<boost::asio::ip::udp::socket>(boost::asio::ip::udp::socket &socket, std::size_t buffer_size);
template void set_socket_recv_buffer_size<boost::asio::ip::tcp::acceptor>(boost::asio::ip::tcp::acceptor &socket, std::size_t buffer_size);
template void set_socket_recv_buffer_size<boost::asio::ip::udp::socket>(boost::asio::ip::udp::socket &socket, std::size_t buffer_size);

} // namespace spead2
