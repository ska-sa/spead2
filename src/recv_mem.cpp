#include <cstdint>
#include <boost/asio.hpp>
#include "recv.h"
#include "recv_mem.h"

namespace spead
{
namespace recv
{

mem_stream::mem_stream(const std::uint8_t *ptr, std::size_t length)
    : ptr(ptr), length(length)
{
    assert(ptr != nullptr);
}

void mem_stream::run()
{
    while (length > 0)
    {
        packet_header packet;
        std::size_t size = decode_packet(packet, ptr, length);
        if (size > 0)
        {
            add_packet(packet);
            ptr += size;
            length -= size;
        }
        else
            length = 0; // causes loop to exit
    }
    flush();
}

} // namespace recv
} // namespace spead
