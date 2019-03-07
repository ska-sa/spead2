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

#ifndef SPEAD2_RECV_UDP_H
#define SPEAD2_RECV_UDP_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#if SPEAD2_USE_RECVMMSG
# include <sys/socket.h>
# include <sys/types.h>
#endif
#include <cstdint>
#include <boost/asio.hpp>
#include <spead2/recv_reader.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp_base.h>

namespace spead2
{
namespace recv
{

/**
 * Asynchronous stream reader that receives packets over UDP.
 */
class udp_reader : public udp_reader_base
{
private:
    /// UDP socket we are listening on
    boost::asio::ip::udp::socket socket;
    /// Unused, but need to provide the memory for asio to write to
    boost::asio::ip::udp::endpoint endpoint;
    /// Maximum packet size we will accept
    std::size_t max_size;
#if SPEAD2_USE_RECVMMSG
    /**
     * A dup(2) of @ref socket. This is used for the actual recvmmsg call. The
     * duplicate is needed so that we can asynchronously call socket.close()
     * to shut down the reader while racing with the recvmmsg call. The
     * close call will just cancel pending handlers, without causing us to
     * read from a closed file descriptor.
     *
     * TODO: probably no longer needed, since stop() is called with the
     * queue_mutex held.
     */
    boost::asio::ip::udp::socket socket2;
    /// Buffer for asynchronous receive, of size @a max_size + 1.
    std::vector<std::unique_ptr<std::uint8_t[]>> buffer;
    /// Scatter-gather array for each buffer
    std::vector<iovec> iov;
    /// recvmmsg control structures
    std::vector<mmsghdr> msgvec;
#else
    /// Buffer for asynchronous receive, of size @a max_size + 1.
    std::unique_ptr<std::uint8_t[]> buffer;
#endif

    /// Start an asynchronous receive
    void enqueue_receive();

    /// Callback on completion of asynchronous receive
    void packet_handler(
        const boost::system::error_code &error,
        std::size_t bytes_transferred);

public:
    /// Socket receive buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 8 * 1024 * 1024;
    /// Number of packets to receive in one go, if recvmmsg support is present
    static constexpr std::size_t mmsg_count = 64;

    /**
     * Constructor.
     *
     * If @a endpoint is a multicast address, then this constructor will
     * subscribe to the multicast group, and also set @c SO_REUSEADDR so that
     * multiple sockets can be subscribed to the multicast group.
     *
     * @param owner        Owning stream
     * @param endpoint     Address on which to listen
     * @param max_size     Maximum packet size that will be accepted.
     * @param buffer_size  Requested socket buffer size. Note that the
     *                     operating system might not allow a buffer size
     *                     as big as the default.
     */
    udp_reader(
        stream &owner,
        const boost::asio::ip::udp::endpoint &endpoint,
        std::size_t max_size = default_max_size,
        std::size_t buffer_size = default_buffer_size);

    /**
     * Constructor with explicit multicast interface address (IPv4 only).
     *
     * The socket will have @c SO_REUSEADDR set, so that multiple sockets can
     * all listen to the same multicast stream. If you want to let the
     * system pick the interface for the multicast subscription, use
     * @c boost::asio::ip::address_v4::any(), or use the default constructor.
     *
     * @param owner        Owning stream
     * @param endpoint     Multicast group and port
     * @param max_size     Maximum packet size that will be accepted.
     * @param buffer_size  Requested socket buffer size.
     * @param interface_address  Address of the interface which should join the group
     *
     * @throws std::invalid_argument If @a endpoint is not an IPv4 multicast address
     * @throws std::invalid_argument If @a interface_address is not an IPv4 address
     */
    udp_reader(
        stream &owner,
        const boost::asio::ip::udp::endpoint &endpoint,
        std::size_t max_size,
        std::size_t buffer_size,
        const boost::asio::ip::address &interface_address);

    /**
     * Constructor with explicit multicast interface index (IPv6 only).
     *
     * The socket will have @c SO_REUSEADDR set, so that multiple sockets can
     * all listen to the same multicast stream. If you want to let the
     * system pick the interface for the multicast subscription, set
     * @a interface_index to 0, or use the standard constructor.
     *
     * @param owner        Owning stream
     * @param endpoint     Multicast group and port
     * @param max_size     Maximum packet size that will be accepted.
     * @param buffer_size  Requested socket buffer size.
     * @param interface_index  Address of the interface which should join the group
     *
     * @see if_nametoindex(3)
     */
    udp_reader(
        stream &owner,
        const boost::asio::ip::udp::endpoint &endpoint,
        std::size_t max_size,
        std::size_t buffer_size,
        unsigned int interface_index);

    /**
     * Constructor using an existing socket. This allows socket options (e.g.,
     * multicast subscriptions) to be fine-tuned by the caller. The socket
     * should not be bound. Note that there is no special handling for
     * multicast addresses here.
     *
     * @deprecated Use the variant taking a pre-bound socket instead.
     *
     * @param owner        Owning stream
     * @param socket       Existing socket which will be taken over. It must
     *                     use the same I/O service as @a owner.
     * @param endpoint     Address on which to listen
     * @param max_size     Maximum packet size that will be accepted.
     * @param buffer_size  Requested socket buffer size. Note that the
     *                     operating system might not allow a buffer size
     *                     as big as the default.
     */
    udp_reader(
        stream &owner,
        boost::asio::ip::udp::socket &&socket,
        const boost::asio::ip::udp::endpoint &endpoint,
        std::size_t max_size = default_max_size,
        std::size_t buffer_size = default_buffer_size);

    /**
     * Constructor using an existing socket. This allows socket options (e.g.,
     * multicast subscriptions) to be fine-tuned by the caller. The socket
     * must already be bound to the desired endpoint. There is no special
     * handling of multicast subscriptions or socket buffer sizes here.
     *
     * @param owner        Owning stream
     * @param socket       Existing socket which will be taken over. It must
     *                     use the same I/O service as @a owner.
     * @param max_size     Maximum packet size that will be accepted.
     */
    udp_reader(
        stream &owner,
        boost::asio::ip::udp::socket &&socket,
        std::size_t max_size = default_max_size);

    virtual void stop() override;
};

/**
 * Factory overload to allow udp_reader to be dynamically substituted with
 * udp_ibv_reader based on environment variables.
 */
template<>
struct reader_factory<udp_reader>
{
    static std::unique_ptr<reader> make_reader(
        stream &owner,
        const boost::asio::ip::udp::endpoint &endpoint,
        std::size_t max_size = udp_reader::default_max_size,
        std::size_t buffer_size = udp_reader::default_buffer_size);

    static std::unique_ptr<reader> make_reader(
        stream &owner,
        const boost::asio::ip::udp::endpoint &endpoint,
        std::size_t max_size,
        std::size_t buffer_size,
        const boost::asio::ip::address &interface_address);

    static std::unique_ptr<reader> make_reader(
        stream &owner,
        const boost::asio::ip::udp::endpoint &endpoint,
        std::size_t max_size,
        std::size_t buffer_size,
        unsigned int interface_index);

    static std::unique_ptr<reader> make_reader(
        stream &owner,
        boost::asio::ip::udp::socket &&socket,
        const boost::asio::ip::udp::endpoint &endpoint,
        std::size_t max_size = udp_reader::default_max_size,
        std::size_t buffer_size = udp_reader::default_buffer_size);

    static std::unique_ptr<reader> make_reader(
        stream &owner,
        boost::asio::ip::udp::socket &&socket,
        std::size_t max_size = udp_reader::default_max_size);
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_UDP_H
