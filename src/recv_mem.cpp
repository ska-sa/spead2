/**
 * @file
 */

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
    boost::asio::io_service &io_service, stream &s,
    const std::uint8_t *ptr, std::size_t length)
    : reader(io_service, s), ptr(ptr), length(length)
{
    assert(ptr != nullptr);
}

void mem_reader::start()
{
    get_io_service().post([this] {
        mem_to_stream(get_stream(), ptr, length);
        // There will be no more data, so we can stop the stream immediately.
        get_stream().stop();
    });
}

} // namespace recv
} // namespace spead
