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
 *
 * Support for netmap.
 */

#ifndef SPEAD2_RECV_NETMAP_UDP_READER_H
#define SPEAD2_RECV_NETMAP_UDP_READER_H

#include <spead2/common_features.h>
#if SPEAD2_USE_NETMAP

#define NETMAP_WITH_LIBS
#include <cstdint>
#include <string>
#include <boost/asio.hpp>
#include <net/netmap_user.h>
#include <spead2/recv_reader.h>
#include <spead2/recv_stream.h>

namespace spead2
{
namespace recv
{

namespace detail
{

class nm_desc_destructor
{
public:
    void operator()(nm_desc *) const;
};

} // namespace detail

class netmap_udp_reader : public reader
{
private:
    /// File handle for the netmap mapping, usable with asio
    boost::asio::posix::stream_descriptor handle;
    /// Information about the netmap mapping
    std::unique_ptr<nm_desc, detail::nm_desc_destructor> desc;
    /// UDP port to listen on
    uint16_t port;

    /// Start an asynchronous receive
    void enqueue_receive();

    /// Callback on completion of asynchronous notification
    void packet_handler(const boost::system::error_code &error);

public:
    /**
     * Constructor.
     *
     * @param owner        Owning stream
     * @param device       Name of the network interface e.g., @c eth0
     * @param port         UDP port number to listen to
     */
    netmap_udp_reader(stream &owner, const std::string &device, uint16_t port);

    virtual void stop() override;
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_NETMAP

#endif // SPEAD2_RECV_NETMAP_UDP_READER_H
