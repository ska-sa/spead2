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

#include <spead2/common_features.h>
#include <system_error>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <boost/asio.hpp>
#include <spead2/recv_reader.h>
#include <spead2/recv_tcp.h>
#include <spead2/common_endian.h>
#include <spead2/common_logging.h>

namespace spead2
{
namespace recv
{

constexpr std::size_t tcp_reader::default_buffer_size;

tcp_reader::tcp_reader(
    stream &owner,
    const boost::asio::ip::tcp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size)
    : udp_reader_base(owner), acceptor(owner.get_strand().get_io_service(), endpoint),
    peer(acceptor.get_io_service()),
    max_size(max_size),
    buffer(new std::uint8_t[max_size * pkts_per_buffer]),
    buffer2(new std::uint8_t[max_size * pkts_per_buffer])
{
    assert(&this->acceptor.get_io_service() == &get_io_service());

    if (buffer_size != 0)
    {
        boost::asio::socket_base::receive_buffer_size option(buffer_size);
        boost::system::error_code ec;
        acceptor.set_option(option, ec);
        if (ec)
        {
            log_warning("request for buffer size %s failed (%s): refer to documentation for details on increasing buffer size",
                        buffer_size, ec.message());
        }
        else
        {
            // Linux silently clips to the maximum allowed size
            boost::asio::socket_base::receive_buffer_size actual;
            acceptor.get_option(actual);
            if (std::size_t(actual.value()) < buffer_size)
            {
                log_warning("requested buffer size %d but only received %d: refer to documentation for details on increasing buffer size",
                            buffer_size, actual.value());
            }
        }
    }

    acceptor.async_accept(this->peer,
        std::bind(&tcp_reader::accept_handler, this, std::placeholders::_1));
}

void tcp_reader::packet_handler(
    const boost::system::error_code &error,
    std::size_t bytes_transferred)
{
    bool read_more = false;
    if (!error)
    {
        if (get_stream_base().is_stopped())
            log_info("TCP reader: discarding packet received after stream stopped");
        else
            read_more = process_buffer(bytes_transferred);
    }
    else if (error != boost::asio::error::operation_aborted)
        log_warning("Error in TCP receiver: %1%", error.message());

    if (read_more)
        enqueue_receive();
    else
    {
        peer.close();
        this->stopped();
    }
}

bool tcp_reader::parse_packet(std::size_t &bytes_avail)
{
    bool stopped = false;

    /* The packet can be fully in the previous buffer,
     * scattered through the previous and current buffer,
     * or fully in the current buffer
     */
    if (buffer2_bytes_avail >= pkt_size)
    {
        stopped = process_one_packet(buffer2.get() + buffer_offset, pkt_size, max_size);
        buffer2_bytes_avail -= pkt_size;
        buffer_offset += pkt_size;
    }
    else if (buffer2_bytes_avail > 0)
    {
        std::uint8_t buf[max_size];
        std::memcpy(buf, buffer2.get() + buffer_offset, buffer2_bytes_avail);
        std::memcpy(buf + buffer2_bytes_avail, buffer.get(), pkt_size - buffer2_bytes_avail);
        stopped = process_one_packet(buf, pkt_size, max_size);
        buffer_offset = pkt_size - buffer2_bytes_avail;
        buffer2_bytes_avail = 0;
    }
    else
    {
        stopped = process_one_packet(buffer.get() + buffer_offset, pkt_size, max_size);
        buffer_offset += pkt_size;
    }

    bytes_avail -= pkt_size;
    pkt_size = 0;
    return stopped;
}

#ifdef SPEAD2_LOG_RECV_TCP
#define LOG_STEP(x) \
    log_debug(x ". bytes_recv = %1%, buffer2_bytes_avail = %2%, bytes_avail = %3%, " \
        "buffer_offset = %4%, pkt_size = %5%, max_size = %6%", \
        bytes_recv, buffer2_bytes_avail, bytes_avail, buffer_offset, \
        pkt_size, max_size)
#else
#define LOG_STEP(x)
#endif

void tcp_reader::finish_buffer_processing(const std::size_t bytes_recv, const std::size_t bytes_avail)
{
    /* If there are still bytes available in buffer2 it means that
     * we couldn't consume a single byte of this->buffer, and therefore the
     * offset refers to this->buffer2. If that's the case then we need to concatenate
     * the contents of the two buffers onto the current buffer (doesn't matter
     * exactly where, but we put it in position 0), and adjust the necessary bits
     * of information
     */
    if (buffer2_bytes_avail > 0)
    {
        // we are not doing this optimally, but it's not happening that often
        LOG_STEP("Couldn't process any received data, accumulating it into buffer");
        std::uint8_t buf[max_size];
        std::memcpy(buf, buffer2.get() + buffer_offset, buffer2_bytes_avail);
        std::memcpy(buf + buffer2_bytes_avail, buffer.get(), bytes_recv);
        std::memcpy(buffer.get(), buf, bytes_recv + buffer2_bytes_avail);
        buffer2_bytes_avail = bytes_avail;
        buffer_offset = 0;
    }
    else
        buffer2_bytes_avail = bytes_recv - buffer_offset;

    /* Swap buffers and prepare for the next round
     * After this swapping buffer2 *might* contain readable data, and
     * buffer is ready to be rewritten
     */
    std::swap(buffer, buffer2);
    if (buffer2_bytes_avail == 0)
        buffer_offset = 0;

}

bool tcp_reader::process_buffer(const std::size_t bytes_recv)
{
    auto bytes_avail = bytes_recv + buffer2_bytes_avail;
    LOG_STEP("Starting to process buffer");

    // No packet is being parsed at the moment, read the next packet size
    if (pkt_size == 0)
    {
        LOG_STEP("Reading next packet size");

        // This is *highly* unlikely to happen, but it could I guess
        if (!parse_packet_size(bytes_avail))
        {
            finish_buffer_processing(bytes_recv, bytes_avail);
            return true;
        }
    }

    while (bytes_avail >= pkt_size)
    {
        LOG_STEP("Trying to read packet");
        if (parse_packet(bytes_avail))
        {
            // we don't enqueue any more reads, as the stream actually finished
            LOG_STEP("Stream finished");
            return false;
        }

        if (!parse_packet_size(bytes_avail))
            break;
    }

    LOG_STEP("Done with buffer processing");
    finish_buffer_processing(bytes_recv, bytes_avail);
    return true;
}

bool tcp_reader::parse_packet_size(std::size_t &bytes_avail)
{
    /* The 8 bytes we need can be fully in the previous buffer,
     * scattered through the previous and current buffer,
     * or fully in the current buffer
     */
    if (buffer2_bytes_avail >= 8)
    {
        pkt_size = load_be<std::uint64_t>(buffer2.get() + buffer_offset);
        buffer2_bytes_avail -= 8;
        buffer_offset += 8;
        bytes_avail -= 8;
    }
    else if (buffer2_bytes_avail > 0)
    {
        std::uint8_t s[8];
        std::memcpy(s, buffer2.get() + buffer_offset, buffer2_bytes_avail);
        std::memcpy(s + buffer2_bytes_avail, buffer.get(), 8 - buffer2_bytes_avail);
        pkt_size = load_be<std::uint64_t>(s);
        buffer_offset = 8 - buffer2_bytes_avail;
        buffer2_bytes_avail = 0;
        bytes_avail -= 8;
    }
    else if (bytes_avail >= 8)
    {
        pkt_size = load_be<std::uint64_t>(buffer.get() + buffer_offset);
        buffer_offset += 8;
        bytes_avail -= 8;
    }
    else
        return false;

    return true;
}

void tcp_reader::accept_handler(const boost::system::error_code &error)
{
    if (!error)
        enqueue_receive();
    else
        log_warning("Error in TCP accept: %1%", error.message());
}

void tcp_reader::enqueue_receive()
{
    using namespace std::placeholders;
    peer.async_receive(
        boost::asio::buffer(buffer.get(), max_size * pkts_per_buffer),
        get_stream().get_strand().wrap(
            std::bind(&tcp_reader::packet_handler,
                this, _1, _2)));
}

void tcp_reader::stop()
{
    /* asio guarantees that closing a socket will cancel any pending
     * operations on it.
     * Don't put any logging here: it could be running in a shutdown
     * path where it is no longer safe to do so.
     */
    acceptor.close();
}

/////////////////////////////////////////////////////////////////////////////

std::unique_ptr<reader> reader_factory<tcp_reader>::make_reader(
    stream &owner,
    const boost::asio::ip::tcp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size)
{
    return std::unique_ptr<reader>(new tcp_reader(owner, endpoint, max_size, buffer_size));
}

} // namespace recv
} // namespace spead2
