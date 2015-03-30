/**
 * @file
 */

#include <cstdint>
#include <cassert>
#include "recv_reader.h"
#include "recv_mem.h"
#include "recv_stream.h"

namespace spead2
{
namespace recv
{

mem_reader::mem_reader(
    stream &owner,
    const std::uint8_t *ptr, std::size_t length)
    : reader(owner), ptr(ptr), length(length)
{
    assert(ptr != nullptr);
    get_stream().get_strand().post([this] {
        mem_to_stream(get_stream_base(), this->ptr, this->length);
        // There will be no more data, so we can stop the stream immediately.
        get_stream_base().stop_received();
        stopped();
    });
}

} // namespace recv
} // namespace spead2
