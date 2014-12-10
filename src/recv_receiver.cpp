#include <memory>
#include "recv.h"
#include "recv_receiver.h"

namespace spead
{
namespace recv
{

void receiver::add_reader(std::unique_ptr<reader> &&r)
{
    reader *ptr = r.get();
    readers.push_back(std::move(r));
    ptr->start(io_service);
}

void receiver::start()
{
    worker = std::async(std::launch::async, [this] {io_service.run();});
}

void receiver::stop()
{
    io_service.stop();
}

void receiver::join()
{
    worker.get();
}

} // namespace recv
} // namespace spead
