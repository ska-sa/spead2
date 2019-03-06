/*
 * TCP receiver for SPEAD protocol
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

#ifndef SPEAD2_RECV_TCP_H
#define SPEAD2_RECV_TCP_H

#include <spead2/common_features.h>
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
 * Asynchronous stream reader that receives packets over TCP.
 */
class tcp_reader : public reader
{
private:
    /// The acceptor object
    boost::asio::ip::tcp::acceptor acceptor;
    /// TCP peer socket (i.e., the one connected to the remote end)
    boost::asio::ip::tcp::socket peer;
    /// Maximum packet size we will accept. Needed mostly for the underlying packet deserialization logic
    std::size_t max_size;
    /// Buffer for packet data reception
    std::unique_ptr<std::uint8_t[]> buffer;
    /// The head of the buffer, data is available from this point up to the tail
    std::uint8_t *head;
    /// The tail of the buffer, after this there is no more data
    std::uint8_t *tail;
    /// Size of the current packet being parsed. 0 means no packet is being parsed
    std::size_t pkt_size = 0;
    /// Number of bytes that need to be skipped (used when pkt_size > max_size)
    std::size_t to_skip = 0;
    /// Number of packets to hold on each buffer for asynchronous receive
    static constexpr std::size_t pkts_per_buffer = 64;

    /// Start an asynchronous receive
    void enqueue_receive();

    /// Callback on completion of asynchronous accept
    void accept_handler(
        const boost::system::error_code &error);

    /// Callback on completion of asynchronous receive
    void packet_handler(
        const boost::system::error_code &error,
        std::size_t bytes_transferred);

    /// Processes the content of the buffer, returns true if more reading needs to be enqueued
    bool process_buffer(stream_base::add_packet_state &state, const std::size_t bytes_recv);

    /// Parses the size of the next packet to read from the stream, returns true if more data needs to be read to parse the packet size correctly
    bool parse_packet_size();

    /// Parses the next packet out of the stream, returns false if the contents of the current stream are not enough
    bool parse_packet(stream_base::add_packet_state &state);

    /// Ignores bytes from the stream according to @a to_skip, returns true if more data needs to be read and skipped
    bool skip_bytes();

    /**
     * Base constructor, used by the other constructors.
     *
     * @param owner        Owning stream
     * @param acceptor     Acceptor object, must be bound
     * @param max_size     Maximum packet size that will be accepted.
     * @param buffer_size  Requested socket buffer size. Note that the
     *                     operating system might not allow a buffer size
     *                     as big as the default.
     */
    tcp_reader(
        stream &owner,
        boost::asio::ip::tcp::acceptor &&acceptor,
        std::size_t max_size,
        std::size_t buffer_size);


public:
    /// Maximum packet size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_max_size = 65536;
    /// Socket receive buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 208 * 1024;

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
    tcp_reader(
        stream &owner,
        const boost::asio::ip::tcp::endpoint &endpoint,
        std::size_t max_size = default_max_size,
        std::size_t buffer_size = default_buffer_size);

    /**
     * Constructor using an existing acceptor object. This allows acceptor objects
     * to be created and fine-tuned by users before handing them over. The
     * acceptor object must be already bound.
     *
     * @param owner        Owning stream
     * @param acceptor     Acceptor object, must be bound
     * @param max_size     Maximum packet size that will be accepted.
     */
    tcp_reader(
        stream &owner,
        boost::asio::ip::tcp::acceptor &&acceptor,
        std::size_t max_size = default_max_size);

    virtual void stop() override;

    virtual bool lossy() const override;

};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_TCP_H
