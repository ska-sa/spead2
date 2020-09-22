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

/**
 * @file
 */

#ifndef SPEAD2_SEND_UDP_H
#define SPEAD2_SEND_UDP_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#if SPEAD2_USE_SENDMMSG
# include <sys/socket.h>
# include <sys/types.h>
#endif
#include <boost/asio.hpp>
#include <utility>
#include <vector>
#include <spead2/send_packet.h>
#include <spead2/send_stream.h>

namespace spead2
{
namespace send
{

class udp_stream : public stream_impl<udp_stream>
{
public:
    /**
     * Helper class for specifying list of destinations.
     *
     * It is not intended to be used directly, but is public as it may be
     * useful in wrappers. It may be removed in a future release. User
     * code should pass either a single endpoint or a vector of endpoints.
     */
    struct destinations
    {
        std::vector<boost::asio::ip::udp::endpoint> endpoints;

        destinations(std::vector<boost::asio::ip::udp::endpoint> endpoints);

        // Templating allows basic_resolver_entry for example
        template<typename T, typename = typename std::enable_if<std::is_convertible<T, boost::asio::ip::udp::endpoint>::value>::type>
        destinations(const T &endpoint)
            : destinations(std::vector<boost::asio::ip::udp::endpoint>{endpoint}) {}
    };

private:
    friend class stream_impl<udp_stream>;
    boost::asio::ip::udp::socket socket;
    std::vector<boost::asio::ip::udp::endpoint> endpoints;

    /// Implements async_send_packets, starting from @a first
    void send_packets(std::size_t first);

    void async_send_packets();

    static constexpr int batch_size = 64;
#if SPEAD2_USE_SENDMMSG
    struct mmsghdr msgvec[batch_size];
    std::vector<struct iovec> msg_iov;
#endif

    /**
     * Constructor used to implement most other destructors.
     */
    udp_stream(
        io_service_ref io_service,
        boost::asio::ip::udp::socket &&socket,
        destinations &&endpoints,
        const stream_config &config,
        std::size_t buffer_size);

public:
    /// Socket send buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 512 * 1024;

    /**
     * Constructor.
     *
     * This constructor can handle unicast or multicast destinations, but is
     * primarily intended for unicast as it does not provide all the options
     * that the multicast-specific constructors do.
     *
     * @param io_service   I/O service for sending data
     * @param endpoints    Destination address and port for each substream
     * @param config       Stream configuration
     * @param buffer_size  Socket buffer size (0 for OS default)
     * @param interface_address   Source address
     *                            @verbatim embed:rst:leading-asterisks
     *                            (see tips on :ref:`routing`)
     *                            @endverbatim
     */
    udp_stream(
        io_service_ref io_service,
        destinations &&endpoints,
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size,
        const boost::asio::ip::address &interface_address = boost::asio::ip::address());

    /**
     * Constructor using an existing socket and an explicit io_service or
     * thread pool. The socket must be open but not connected, and the
     * io_service must match the socket's.
     */
    udp_stream(
        io_service_ref io_service,
        boost::asio::ip::udp::socket &&socket,
        destinations &&endpoints,
        const stream_config &config = stream_config());

    /**
     * Constructor with multicast hop count.
     *
     * @param io_service   I/O service for sending data
     * @param endpoints    Multicast group and port for each substream
     * @param config       Stream configuration
     * @param buffer_size  Socket buffer size (0 for OS default)
     * @param ttl          Maximum number of hops
     *
     * @throws std::invalid_argument if @a endpoint is not a multicast address
     */
    udp_stream(
        io_service_ref io_service,
        destinations &&endpoints,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl);

    /**
     * Constructor with multicast hop count and outgoing interface address
     * (IPv4 only).
     *
     * @param io_service   I/O service for sending data
     * @param endpoints    Multicast group and port for each substream
     * @param config       Stream configuration
     * @param buffer_size  Socket buffer size (0 for OS default)
     * @param ttl          Maximum number of hops
     * @param interface_address   Address of the outgoing interface
     *
     * @throws std::invalid_argument if @a endpoint is not an IPv4 multicast address
     * @throws std::invalid_argument if @a interface_address is not an IPv4 address
     */
    udp_stream(
        io_service_ref io_service,
        destinations &&endpoints,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        const boost::asio::ip::address &interface_address);

    /**
     * Constructor with multicast hop count and outgoing interface address
     * (IPv6 only).
     *
     * @param io_service   I/O service for sending data
     * @param endpoints    Multicast group and port for each substream
     * @param config       Stream configuration
     * @param buffer_size  Socket buffer size (0 for OS default)
     * @param ttl          Maximum number of hops
     * @param interface_index   Index of the outgoing interface
     *
     * @throws std::invalid_argument if @a endpoint is not an IPv6 multicast address
     *
     * @see if_nametoindex(3)
     */
    udp_stream(
        io_service_ref io_service,
        destinations &&endpoints,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        unsigned int interface_index);

    std::size_t get_num_substreams() const { return endpoints.size(); }

    virtual ~udp_stream();
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_UDP_H
