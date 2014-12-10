#include <cstdint>
#include <functional>
#include <boost/asio.hpp>
#include "recv_reader.h"
#include "recv_udp.h"

namespace spead
{
namespace recv
{

constexpr std::size_t udp_reader::default_max_size;

udp_reader::udp_reader(
    stream *s,
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size)
    : reader(s), socket(io_service), max_size(max_size)
{
    // Allocate one extra byte so that overflow can be detected
    buffer.reset(new std::uint8_t[max_size + 1]);
    socket.open(endpoint.protocol());
    if (buffer_size != 0)
    {
        boost::asio::socket_base::receive_buffer_size option(buffer_size);
        socket.set_option(option);
    }
    socket.bind(endpoint);
}

void udp_reader::packet_handler(
    const boost::system::error_code &error,
    std::size_t bytes_transferred)
{
    // TODO check error
    // If it's bigger, the packet might have been truncated
    if (bytes_transferred <= max_size && bytes_transferred > 0)
    {
        packet_header packet;
        std::size_t size = decode_packet(packet, buffer.get(), bytes_transferred);
        if (size == bytes_transferred)
            get_stream()->add_packet(packet);
    }

    if (socket.is_open())
    {
        using namespace std::placeholders;
        // TODO: check error
        socket.async_receive_from(
            boost::asio::buffer(buffer.get(), max_size + 1),
            endpoint,
            std::bind(&udp_reader::packet_handler, this, _1, _2));
    }
}

void udp_reader::start(boost::asio::io_service &)
{
    using namespace std::placeholders;
    // TODO: check error
    socket.async_receive_from(
        boost::asio::buffer(buffer.get(), max_size + 1),
        endpoint,
        std::bind(&udp_reader::packet_handler, this, _1, _2));
}

void udp_reader::stop()
{
    reader::stop();
    socket.close();
}

} // namespace recv
} // namespace spead
