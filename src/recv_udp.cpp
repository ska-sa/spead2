/**
 * @file
 */

#include <cstdint>
#include <functional>
#include <boost/asio.hpp>
#include <iostream>
#include "recv_reader.h"
#include "recv_udp.h"
#include "common_logging.h"

namespace spead
{
namespace recv
{

constexpr std::size_t udp_reader::default_max_size;
constexpr std::size_t udp_reader::default_buffer_size;
constexpr int udp_reader::slots;

udp_reader::udp_reader(
    boost::asio::io_service &io_service,
    stream &s,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size)
    : reader(io_service, s), socket(io_service), max_size(max_size)
{
    for (int i = 0; i < slots; i++)
    {
        // Allocate one extra byte so that overflow can be detected
        buffers[i].reset(new std::uint8_t[max_size + 1]);
    }
    socket.open(endpoint.protocol());
    if (buffer_size != 0)
    {
        boost::asio::socket_base::receive_buffer_size option(buffer_size);
        socket.set_option(option);
        boost::asio::socket_base::receive_buffer_size actual;
        socket.get_option(actual);
        if (std::size_t(actual.value()) < buffer_size)
        {
            log_warning("requested buffer size %d but only received %d: refer to documentation for details on increasing buffer size",
                        buffer_size, actual.value());
        }
    }
    socket.bind(endpoint);
}

void udp_reader::packet_handler(
    int phase,
    const boost::system::error_code &error,
    std::size_t bytes_transferred)
{
    if (!get_stream().is_stopped())
    {
        /* Start the next async receive immediately, while we handle the packet.
         * It might still be cancelled if the current packet stops the stream.
         * It can also cause an additional callback if a new packet arrives
         * before the stream is stopped.
         */
        enqueue_receive(1 - phase);
    }

    if (!error)
    {
        if (get_stream().is_stopped())
        {
            log_debug("UDP reader: discarding packet received after stream stopped");
        }
        else if (bytes_transferred <= max_size && bytes_transferred > 0)
        {
            const std::uint8_t *buffer = buffers[phase].get();
            // If it's bigger, the packet might have been truncated
            packet_header packet;
            std::size_t size = decode_packet(packet, buffer, bytes_transferred);
            if (size == bytes_transferred)
            {
                get_stream().add_packet(packet);
                if (get_stream().is_stopped())
                {
                    log_debug("UDP reader: end of stream detected");
                    socket.close();
                }
            }
        }
        else if (bytes_transferred > max_size)
            log_debug("dropped packet due to truncation");
    }
    // TODO: log the error if there was one
}

void udp_reader::enqueue_receive(int phase)
{
    using namespace std::placeholders;
    socket.async_receive_from(
        boost::asio::buffer(buffers[phase].get(), max_size + 1),
        endpoints[phase],
        get_strand().wrap(std::bind(&udp_reader::packet_handler, this, phase, _1, _2)));
}

void udp_reader::start()
{
    enqueue_receive(0);
}

void udp_reader::stop()
{
    // Don't put any logging here: it could be running in a shutdown
    // path where it is no longer safe to do so
    reader::stop();
    // asio guarantees that closing a socket will cancel any pending
    // operations on it
    socket.close();
}

} // namespace recv
} // namespace spead
