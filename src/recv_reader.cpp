/**
 * @file
 */

#include "recv_reader.h"
#include "recv_stream.h"

namespace spead
{
namespace recv
{

boost::asio::io_service &reader::get_io_service()
{
    return owner.get_strand().get_io_service();
}

} // namespace recv
} // namespace spead
