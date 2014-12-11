/**
 * @file
 */

#include <memory>
#include <thread>
#include "recv_receiver.h"

namespace spead
{
namespace recv
{

void receiver::operator()()
{
    if (worker.valid())
        throw std::invalid_argument("Receiver is already running");
    std::promise<void> promise;
    worker = promise.get_future();
    io_service.run();
    promise.set_value(); // tells stop() that we have terminated
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
