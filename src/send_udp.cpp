#include <cstddef>
#include <utility>
#include <boost/asio.hpp>
#include "send_udp.h"
#include "common_defines.h"

namespace spead
{
namespace send
{

constexpr std::size_t udp_stream::default_buffer_size;

udp_stream::udp_stream(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    int heap_address_bits,
    std::size_t max_packet_size,
    double rate,
    std::size_t max_heaps,
    std::size_t buffer_size)
    : stream<udp_stream>(io_service, heap_address_bits,
                         max_packet_size, rate, max_heaps),
    socket(io_service), endpoint(endpoint)
{
    boost::asio::ip::udp::socket socket(io_service);
    socket.open(endpoint.protocol());
    if (buffer_size != 0)
    {
        boost::asio::socket_base::send_buffer_size option(buffer_size);
        socket.set_option(option);
        boost::asio::socket_base::send_buffer_size actual;
        socket.get_option(actual);
        if (std::size_t(actual.value()) < buffer_size)
        {
            log_warning("requested buffer size %d but only received %d: refer to documentation for details on increasing buffer size",
                        buffer_size, actual.value());
        }
    }
    this->socket = std::move(socket);
}

} // namespace send
} // namespace spead
