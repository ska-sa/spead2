#include <memory>
#include "in.h"
#include "receiver.h"

namespace spead
{
namespace in
{

void receiver::add_stream(std::unique_ptr<stream> &&s)
{
    stream *ptr = s.get();
    streams.push_back(std::move(s));
    ptr->start();
}

} // namespace in
} // namespace spead
