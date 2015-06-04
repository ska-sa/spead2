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

#include <cstdint>
#include <boost/asio.hpp>
#include "recv_reader.h"
#include "recv_stream.h"

namespace spead2
{
namespace recv
{

/**
 * Asynchronous stream reader that receives packets over UDP.
 *
 * @todo Log errors somehow?
 */
class udp_reader : public reader
{
private:
    /// UDP socket we are listening on
    boost::asio::ip::udp::socket socket;
    /// Unused, but need to provide the memory for asio to write to
    boost::asio::ip::udp::endpoint endpoint;
    /// Buffer for asynchronous receive, of size @a max_size + 1.
    std::unique_ptr<std::uint8_t[]> buffer;
    /// Maximum packet size we will accept
    std::size_t max_size;

    /// Start an asynchronous receive
    void enqueue_receive();

    /// Callback on completion of asynchronous receive
    void packet_handler(
        const boost::system::error_code &error,
        std::size_t bytes_transferred);

public:
    /// Maximum packet size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_max_size = 9200;
    /// Socket receive buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 8 * 1024 * 1024;

    /**
     * Constructor.
     *
     * @param owner        Owning stream
     * @param endpoint     Address on which to listen
     * @param max_size     Maximum packet size that will be accepted.
     * @param buffer_size  Requested socket buffer size. Note that the
     *                     operating system might not allow a buffer size
     *                     as big as the default.
     */
    explicit udp_reader(
        stream &owner,
        const boost::asio::ip::udp::endpoint &endpoint,
        std::size_t max_size = default_max_size,
        std::size_t buffer_size = default_buffer_size);

    virtual void stop() override;
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_UDP_H
