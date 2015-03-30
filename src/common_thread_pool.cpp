/**
 * @file
 */

#include <memory>
#include <thread>
#include <stdexcept>
#include "common_thread_pool.h"
#include "common_logging.h"

namespace spead2
{

thread_pool::thread_pool(int num_threads)
    : work(io_service)
{
    if (num_threads < 1)
        throw std::invalid_argument("at least one thread is required");
    workers.reserve(num_threads);
    for (int i = 0; i < num_threads; i++)
    {
        workers.push_back(std::async(std::launch::async, [this] {io_service.run();}));
    }
}

void thread_pool::stop()
{
    io_service.stop();
    for (auto &worker : workers)
    {
        try
        {
            worker.get();
        }
        catch (std::exception &e)
        {
            log_warning("worker thread throw an exception: %s", e.what());
        }
    }
    workers.clear();
}

thread_pool::~thread_pool()
{
    stop();
}

} // namespace spead2
