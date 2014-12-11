/**
 * @file
 */

#include "recv_reader.h"
#include "recv_stream.h"

namespace spead
{
namespace recv
{

void reader::stop()
{
    s.stop();
}

} // namespace recv
} // namespace spead
