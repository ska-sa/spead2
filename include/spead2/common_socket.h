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

#ifndef SPEAD2_COMMON_SOCKET_H
#define SPEAD2_COMMON_SOCKET_H

#include <boost/asio.hpp>

namespace spead2
{

/**
 * Tries to set the socket's send buffer size option and warns if it cannot be done.
 * If @a buffer_size is zero the socket is left unchanged.
 *
 * @param socket The socket on which the send buffer size will be set
 * @param buffer_size The buffer size
 */
template<typename SocketType>
void set_socket_send_buffer_size(SocketType &socket, std::size_t buffer_size);

/**
 * Tries to set the socket's receive buffer size option and warns if it cannot be done.
 * If @a buffer_size is zero the socket is left unchanged.
 *
 * @param socket The socket on which the receive buffer size will be set
 * @param buffer_size The buffer size
 */
template<typename SocketType>
void set_socket_recv_buffer_size(SocketType &socket, std::size_t buffer_size);

#if BOOST_VERSION >= 107000
/**
 * Get an object suitable for constructing another socket with the same IO
 * context. The return type depends on the Boost version. This is necessary
 * because Boost 1.70 removed @c basic_socket::get_io_service (as well as
 * @c basic_socket::get_io_context).
 */
template<typename SocketType>
static inline typename SocketType::executor_type get_socket_io_service(SocketType &socket)
{
    return socket.get_executor();
}
#else
template<typename SocketType>
static inline boost::asio::io_service &get_socket_io_service(SocketType &socket)
{
    return socket.get_io_service();
}
#endif

/**
 * Determine whether a socket is using a particular IO context. This is necessary
 * because Boost 1.70 removed @c basic_socket::get_io_service (as well as
 * @c basic_socket::get_io_context).
 */
template<typename SocketType>
static inline bool socket_uses_io_service(SocketType &socket, boost::asio::io_service &io_service)
{
#if BOOST_VERSION >= 107000
    return socket.get_executor() == io_service.get_executor();
#else
    return &socket.get_io_service() == &io_service;
#endif
}

}  // namespace spead2

#endif // SPEAD2_COMMON_SOCKET_H
