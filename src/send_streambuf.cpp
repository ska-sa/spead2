/**
 * @file
 */

#include <streambuf>
#include "send_streambuf.h"

namespace spead2
{
namespace send
{

streambuf_stream::streambuf_stream(
    boost::asio::io_service &io_service,
    std::streambuf &streambuf,
    const stream_config &config)
    : stream<streambuf_stream>(io_service, config), streambuf(streambuf)
{
}

} // namespace send
} // namespace spead2
