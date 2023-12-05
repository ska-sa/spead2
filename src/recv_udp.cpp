/* Copyright 2015, 2019-2020, 2023 National Research Foundation (SARAO)
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

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#if SPEAD2_USE_RECVMMSG
# include <sys/socket.h>
# include <sys/types.h>
# include <unistd.h>
#endif
#include <system_error>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <functional>
#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp.h>
#include <spead2/recv_udp_base.h>
#include <spead2/recv_udp_ibv.h>
#include <spead2/common_logging.h>
#include <spead2/common_socket.h>

namespace spead2::recv
{

udp_reader::udp_reader(
    stream &owner,
    boost::asio::ip::udp::socket &&socket,
    std::size_t max_size)
    : udp_reader_base(owner), max_size(max_size),
#if SPEAD2_USE_RECVMMSG
    buffer(mmsg_count), iov(mmsg_count), msgvec(mmsg_count),
#else
    buffer(new std::uint8_t[max_size + 1]),
#endif
    socket(std::move(socket))
{
    assert(socket_uses_io_service(this->socket, get_io_service()));
#if SPEAD2_USE_RECVMMSG
    for (std::size_t i = 0; i < mmsg_count; i++)
    {
        // Allocate one extra byte so that overflow can be detected
        buffer[i].reset(new std::uint8_t[max_size + 1]);
        iov[i].iov_base = (void *) buffer[i].get();
        iov[i].iov_len = max_size + 1;
        std::memset(&msgvec[i], 0, sizeof(msgvec[i]));
        msgvec[i].msg_hdr.msg_iov = &iov[i];
        msgvec[i].msg_hdr.msg_iovlen = 1;
    }
#endif
}

void udp_reader::start()
{
    if (bind_endpoint)
        socket.bind(*bind_endpoint);
    enqueue_receive(make_handler_context());
}

static boost::asio::ip::udp::socket make_v4_socket(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
{
    if (!interface_address.is_v4())
        throw std::invalid_argument("interface address is not an IPv4 address");
    auto ep = endpoint;
    if (ep.address().is_unspecified())
        ep.address(interface_address);
    if (!ep.address().is_v4())
        throw std::invalid_argument("endpoint is not an IPv4 address");
    if (!ep.address().is_multicast() && ep.address() != interface_address)
        throw std::invalid_argument("endpoint is not multicast and does not match interface address");
    boost::asio::ip::udp::socket socket(io_service, ep.protocol());
    if (ep.address().is_multicast())
    {
        socket.set_option(boost::asio::socket_base::reuse_address(true));
        socket.set_option(boost::asio::ip::multicast::join_group(
            ep.address().to_v4(), interface_address.to_v4()));
    }
    set_socket_recv_buffer_size(socket, buffer_size);
    return socket;
}

static boost::asio::ip::udp::socket make_multicast_v6_socket(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t buffer_size,
    unsigned int interface_index)
{
    if (!endpoint.address().is_v6() || !endpoint.address().is_multicast())
        throw std::invalid_argument("endpoint is not an IPv6 multicast address");
    boost::asio::ip::udp::socket socket(io_service, endpoint.protocol());
    socket.set_option(boost::asio::socket_base::reuse_address(true));
    socket.set_option(boost::asio::ip::multicast::join_group(
        endpoint.address().to_v6(), interface_index));
    set_socket_recv_buffer_size(socket, buffer_size);
    return socket;
}

static boost::asio::ip::udp::socket make_socket(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t buffer_size)
{
    boost::asio::ip::udp::socket socket(io_service, endpoint.protocol());
    if (endpoint.address().is_multicast())
    {
        socket.set_option(boost::asio::socket_base::reuse_address(true));
        socket.set_option(boost::asio::ip::multicast::join_group(endpoint.address()));
    }
    set_socket_recv_buffer_size(socket, buffer_size);
    return socket;
}

udp_reader::udp_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size)
    : udp_reader(
        owner,
        make_socket(owner.get_io_service(), endpoint, buffer_size),
        max_size)
{
    bind_endpoint = endpoint;
}

udp_reader::udp_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
    : udp_reader(
        owner,
        make_v4_socket(owner.get_io_service(),
                       endpoint, buffer_size, interface_address),
        max_size)
{
    auto ep = endpoint;
    // Match the logic in make_v4_socket
    if (ep.address().is_unspecified())
        ep.address(interface_address);
    bind_endpoint = ep;
}

udp_reader::udp_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size,
    unsigned int interface_index)
    : udp_reader(
        owner,
        make_multicast_v6_socket(owner.get_io_service(),
                                 endpoint, buffer_size, interface_index),
        max_size)
{
    bind_endpoint = endpoint;
}

void udp_reader::packet_handler(
    handler_context ctx,
    stream_base::add_packet_state &state,
    const boost::system::error_code &error,
    [[maybe_unused]] std::size_t bytes_transferred)
{
    if (!error)
    {
#if SPEAD2_USE_RECVMMSG
        int received = recvmmsg(socket.native_handle(), msgvec.data(), msgvec.size(),
                                MSG_DONTWAIT, nullptr);
        log_debug("recvmmsg returned %1%", received);
        if (received == -1 && errno != EAGAIN && errno != EWOULDBLOCK)
        {
            std::error_code code(errno, std::system_category());
            log_warning("recvmmsg failed: %1% (%2%)", code.value(), code.message());
        }
        for (int i = 0; i < received; i++)
        {
            bool stopped = process_one_packet(state,
                                              buffer[i].get(), msgvec[i].msg_len, max_size);
            if (stopped)
                break;
        }
#else
        process_one_packet(state, buffer.get(), bytes_transferred, max_size);
#endif
    }
    else if (error != boost::asio::error::operation_aborted)
        log_warning("Error in UDP receiver: %1%", error.message());

    if (!state.is_stopped())
    {
        enqueue_receive(std::move(ctx));
    }
}

void udp_reader::enqueue_receive(handler_context ctx)
{
    using namespace std::placeholders;
#if SPEAD2_USE_RECVMMSG
    socket.async_wait(
        socket.wait_read,
        bind_handler(std::move(ctx), std::bind(&udp_reader::packet_handler, this, _1, _2, _3, 0))
    );
#else
    socket.async_receive_from(
        boost::asio::buffer(buffer.get(), max_size + 1),
        sender_endpoint,
        bind_handler(std::move(ctx), std::bind(&udp_reader::packet_handler, this, _1, _2, _3, _4))
    );
#endif
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

/////////////////////////////////////////////////////////////////////////////

static bool ibv_override;
#if SPEAD2_USE_IBV
static int ibv_comp_vector;
#endif
static boost::asio::ip::address ibv_interface;
static std::once_flag ibv_once;

static void init_ibv_override()
{
    const char *interface = getenv("SPEAD2_IBV_INTERFACE");
    ibv_override = false;
    if (interface && interface[0])
    {
#if !SPEAD2_USE_IBV
        log_warning("SPEAD2_IBV_INTERFACE found, but ibverbs support not compiled in");
#else
        boost::system::error_code ec;
        ibv_interface = boost::asio::ip::address_v4::from_string(interface, ec);
        if (ec)
        {
            log_warning("SPEAD2_IBV_INTERFACE could not be parsed as an IPv4 address: %1%", ec.message());
        }
        else
        {
            ibv_override = true;
            const char *comp_vector = getenv("SPEAD2_IBV_COMP_VECTOR");
            if (comp_vector && comp_vector[0])
            {
                try
                {
                    ibv_comp_vector = boost::lexical_cast<int>(comp_vector);
                }
                catch (boost::bad_lexical_cast &)
                {
                    log_warning("SPEAD2_IBV_COMP_VECTOR is not a valid integer, ignoring");
                }
            }
        }
#endif
    }
}

std::unique_ptr<reader> reader_factory<udp_reader>::make_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size)
{
    if (endpoint.address().is_v4())
    {
        std::call_once(ibv_once, init_ibv_override);
#if SPEAD2_USE_IBV
        if (ibv_override)
        {
            log_info("Overriding reader for %1%:%2% to use ibverbs",
                     endpoint.address().to_string(), endpoint.port());
            return reader_factory<udp_ibv_reader>::make_reader(
                owner,
                udp_ibv_config()
                    .add_endpoint(endpoint)
                    .set_interface_address(ibv_interface)
                    .set_max_size(max_size)
                    .set_buffer_size(buffer_size)
                    .set_comp_vector(ibv_comp_vector));
        }
#endif
    }
    return std::make_unique<udp_reader>(owner, endpoint, max_size, buffer_size);
}

std::unique_ptr<reader> reader_factory<udp_reader>::make_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
{
    if (endpoint.address().is_v4())
    {
        std::call_once(ibv_once, init_ibv_override);
#if SPEAD2_USE_IBV
        if (ibv_override)
        {
            log_info("Overriding reader for %1%:%2% to use ibverbs",
                     endpoint.address().to_string(), endpoint.port());
            return reader_factory<udp_ibv_reader>::make_reader(
                owner,
                udp_ibv_config()
                    .add_endpoint(endpoint)
                    .set_interface_address(interface_address)
                    .set_max_size(max_size)
                    .set_buffer_size(buffer_size)
                    .set_comp_vector(ibv_comp_vector));
        }
#endif
    }
    return std::make_unique<udp_reader>(owner, endpoint, max_size, buffer_size, interface_address);
}

std::unique_ptr<reader> reader_factory<udp_reader>::make_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size,
    unsigned int interface_index)
{
    return std::make_unique<udp_reader>(owner, endpoint, max_size, buffer_size, interface_index);
}

std::unique_ptr<reader> reader_factory<udp_reader>::make_reader(
    stream &owner,
    boost::asio::ip::udp::socket &&socket,
    std::size_t max_size)
{
    return std::make_unique<udp_reader>(owner, std::move(socket), max_size);
}

} // namespace spead2::recv
