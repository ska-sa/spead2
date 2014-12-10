#include <cstdint>
#include <cassert>
#include "recv_reader.h"
#include "recv_mem.h"
#include "recv_stream.h"

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

void mem_reader::start(boost::asio::io_service &io_service)
{
    io_service.post([this] {
        mem_to_stream(*get_stream(), ptr, length);
        get_stream()->end_of_stream();
    });
}

} // namespace recv
} // namespace spead
