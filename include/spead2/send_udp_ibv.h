/* Copyright 2016, 2019-2020 SKA South Africa
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

#ifndef SPEAD2_SEND_UDP_IBV_H
#define SPEAD2_SEND_UDP_IBV_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#if SPEAD2_USE_IBV

#include <boost/asio.hpp>
#include <vector>
#include <initializer_list>
#include <spead2/send_stream.h>
#include <spead2/common_thread_pool.h>

namespace spead2
{
namespace send
{

class udp_ibv_stream : public stream
{
public:
    /// Default send buffer size, if none is passed to the constructor
    static constexpr std::size_t default_buffer_size = 512 * 1024;
    /// Number of times to poll in a row, if none is explicitly passed to the constructor
    static constexpr int default_max_poll = 10;

    /**
     * Backwards-compatibility constructor (taking only a single endpoint).
     *
     * @param io_service   I/O service for sending data
     * @param endpoint     Multicast group and port
     * @param config       Stream configuration
     * @param interface_address   Address of the outgoing interface
     * @param buffer_size  Socket buffer size (0 for OS default)
     * @param ttl          Maximum number of hops
     * @param comp_vector  Completion channel vector (interrupt) for asynchronous operation, or
     *                     a negative value to poll continuously. Polling
     *                     should not be used if there are other users of the
     *                     thread pool. If a non-negative value is provided, it
     *                     is taken modulo the number of available completion
     *                     vectors. This allows a number of readers to be
     *                     assigned sequential completion vectors and have them
     *                     load-balanced, without concern for the number
     *                     available.
     * @param max_poll     Maximum number of times to poll in a row, without
     *                     waiting for an interrupt (if @a comp_vector is
     *                     non-negative) or letting other code run on the
     *                     thread (if @a comp_vector is negative).
     *
     * @throws std::invalid_argument if @a endpoint is not an IPv4 multicast address
     * @throws std::invalid_argument if @a interface_address is not an IPv4 address
     */
    SPEAD2_DEPRECATED("use a vector of endpoints")
    udp_ibv_stream(
        io_service_ref io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        const boost::asio::ip::address &interface_address,
        std::size_t buffer_size = default_buffer_size,
        int ttl = 1,
        int comp_vector = 0,
        int max_poll = default_max_poll);

    /**
     * Constructor.
     *
     * @param io_service   I/O service for sending data
     * @param endpoints    Multicast groups and ports
     * @param config       Stream configuration
     * @param interface_address   Address of the outgoing interface
     * @param buffer_size  Socket buffer size (0 for OS default)
     * @param ttl          Maximum number of hops
     * @param comp_vector  Completion channel vector (interrupt) for asynchronous operation, or
     *                     a negative value to poll continuously. Polling
     *                     should not be used if there are other users of the
     *                     thread pool. If a non-negative value is provided, it
     *                     is taken modulo the number of available completion
     *                     vectors. This allows a number of readers to be
     *                     assigned sequential completion vectors and have them
     *                     load-balanced, without concern for the number
     *                     available.
     * @param max_poll     Maximum number of times to poll in a row, without
     *                     waiting for an interrupt (if @a comp_vector is
     *                     non-negative) or letting other code run on the
     *                     thread (if @a comp_vector is negative).
     *
     * @throws std::invalid_argument if @a endpoint is not an IPv4 multicast address
     * @throws std::invalid_argument if @a interface_address is not an IPv4 address
     */
    udp_ibv_stream(
        io_service_ref io_service,
        const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
        const stream_config &config,
        const boost::asio::ip::address &interface_address,
        std::size_t buffer_size = default_buffer_size,
        int ttl = 1,
        int comp_vector = 0,
        int max_poll = default_max_poll);

    /* Force an initializer list to forward to the vector version (without this,
     * a singleton initializer list forwards to the scalar version).
     */
    udp_ibv_stream(
        io_service_ref io_service,
        std::initializer_list<boost::asio::ip::udp::endpoint> endpoints,
        const stream_config &config,
        const boost::asio::ip::address &interface_address,
        std::size_t buffer_size = default_buffer_size,
        int ttl = 1,
        int comp_vector = 0,
        int max_poll = default_max_poll);
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_USE_IBV
#endif // SPEAD2_SEND_UDP_IBV_H
