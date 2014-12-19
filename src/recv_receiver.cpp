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

receiver::~receiver()
{
    if (!workers.empty())
        stop();
}

void receiver::operator()()
{
    if (!workers.empty())
        throw std::invalid_argument("Receiver is already running");
    std::promise<void> promise;
    workers.push_back(promise.get_future());
    io_service.run();
    promise.set_value(); // tells stop() that we have terminated
}

void receiver::start(int num_threads)
{
    if (!workers.empty())
        throw std::invalid_argument("Receiver is already running");
    workers.reserve(num_threads);
    for (int i = 0; i < num_threads; i++)
    {
        workers.push_back(std::async(std::launch::async, [this] {io_service.run();}));
    }
}

void receiver::stop()
{
    if (workers.empty())
        throw std::invalid_argument("Receiver is not running");
    for (auto &reader : readers)
    {
        reader->get_strand().post([&reader] { reader->stop(); });
    }
    // TODO: if one of the workers threw an exception, we should still wait for
    // all of them, then get the exception
    for (auto &worker : workers)
        worker.get();
    workers.clear();
    readers.clear();
}

} // namespace recv
} // namespace spead
