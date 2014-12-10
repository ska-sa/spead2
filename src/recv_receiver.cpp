#include <memory>
#include "recv.h"
#include "recv_receiver.h"

namespace spead
{
namespace recv
{

void receiver::add_stream(std::unique_ptr<stream> &&s)
{
    stream *ptr = s.get();
    streams.push_back(std::move(s));
    ptr->start();
}

} // namespace recv
} // namespace spead
