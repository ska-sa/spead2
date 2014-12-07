#include <cstdint>
#include <boost/asio.hpp>
#include "defines.h"
#include "in.h"
#include "receiver.h"
#include "udp_in.h"

namespace spead
{
namespace in
{

constexpr std::size_t udp_stream::default_max_size;

udp_stream::udp_stream(
    receiver<udp_stream> &rec,
    const boost::asio::ip::udp::endpoint &endpoint,
    std::size_t max_size,
    std::size_t buffer_size)
    : rec(rec), socket(rec.io_service), max_size(max_size)
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

void udp_stream::ready_handler(
    const boost::system::error_code &error,
    std::size_t bytes_transferred)
{
    using namespace std::placeholders;
    // TODO: check error
    socket.async_receive_from(
        boost::asio::buffer(buffer.get(), max_size + 1),
        endpoint,
        std::bind(&udp_stream::packet_handler, this, _1, _2));
}

void udp_stream::packet_handler(
    const boost::system::error_code &error,
    std::size_t bytes_transferred)
{
    // TODO check error
    if (bytes_transferred <= max_size && bytes_transferred > 0)
    {
        // If it's bigger, the packet might have been truncated
        add_packet(buffer.get(), bytes_transferred);
    }
    start();
}

void udp_stream::start()
{
    using namespace std::placeholders;
    socket.async_receive_from(
        boost::asio::null_buffers(),
        endpoint,
        std::bind(&udp_stream::ready_handler, this, _1, _2));
}

} // namespace in
} // namespace spead
