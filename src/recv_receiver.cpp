#include <memory>
#include <thread>
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
    if (worker.valid())
        throw std::invalid_argument("Receiver is already running");
    worker = std::async(std::launch::async, [this] {io_service.run();});
}

void receiver::stop()
{
    if (!worker.valid())
        throw std::invalid_argument("Receiver is not running");
    io_service.post([this] {
        for (const auto &reader : readers)
        {
            reader->stop();
        }
    });
    worker.get();
    readers.clear();
}

} // namespace recv
} // namespace spead
