#include <cstdint>
#include <cassert>
#include "recv.h"
#include "recv_reader.h"
#include "recv_mem.h"

namespace spead
{
namespace recv
{

mem_reader::mem_reader(
    stream *s,
    const std::uint8_t *ptr, std::size_t length)
    : reader(s), ptr(ptr), length(length)
{
    assert(ptr != nullptr);
}

void mem_reader::run()
{
    while (length > 0)
    {
        packet_header packet;
        std::size_t size = decode_packet(packet, ptr, length);
        if (size > 0)
        {
            get_stream()->add_packet(packet);
            ptr += size;
            length -= size;
        }
        else
            length = 0; // causes loop to exit
    }
    get_stream()->end_of_stream();
}

void mem_reader::start(boost::asio::io_service &io_service)
{
    io_service.post([this] { run(); });
}

} // namespace recv
} // namespace spead
