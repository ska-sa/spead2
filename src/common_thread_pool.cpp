/* Copyright 2015 SKA South Africa
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
