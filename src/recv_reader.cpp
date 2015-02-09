/**
 * @file
 */

#include "recv_reader.h"
#include "recv_stream.h"

namespace spead
{
namespace recv
{

void reader::stopped()
{
    stopped_promise.set_value();
}

boost::asio::io_service &reader::get_io_service()
{
    return owner.get_strand().get_io_service();
}

void reader::join()
{
    stopped_promise.get_future().get();
}

} // namespace recv
} // namespace spead
