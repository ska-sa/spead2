/**
 * @file
 */

#include <streambuf>
#include "send_streambuf.h"

namespace spead
{
namespace send
{

streambuf_stream::streambuf_stream(
    boost::asio::io_service &io_service,
    std::streambuf &streambuf,
    int heap_address_bits,
    std::size_t max_packet_size,
    double rate,
    std::size_t max_heaps)
    : stream<streambuf_stream>(
        io_service, heap_address_bits, max_packet_size, rate, max_heaps),
        streambuf(streambuf)
{
}

} // namespace send
} // namespace spead
