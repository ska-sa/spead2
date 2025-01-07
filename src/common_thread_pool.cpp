/* Copyright 2015-2017, 2019-2020, 2025 National Research Foundation (SARAO)
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
#include <system_error>
#include <spead2/common_thread_pool.h>
#include <spead2/common_logging.h>
#include <spead2/common_features.h>
#if SPEAD2_USE_PTHREAD_SETAFFINITY_NP
# include <sched.h>
# include <pthread.h>
#endif

namespace spead2
{

static void run_io_context(boost::asio::io_context &io_context)
{
    try
    {
        /* Glibc's memory allocator seems to do some initialisation the
         * first time a thread does a dynamic allocation. To avoid that
         * occurring in a real-time situation, do a dummy allocation now to
         * initialise it.
         */
        int *dummy = new int;
        delete dummy;
        io_context.run();
    }
    catch (const std::exception &e)
    {
        log_warning("Worker thread threw exception (expect deadlocks!): %1%", e.what());
        throw;
    }
    catch (...)
    {
        log_warning("Worker thread threw unknown exception (expect deadlocks!)");
        throw;
    }
}

thread_pool::thread_pool(int num_threads)
    : work_guard(boost::asio::make_work_guard(io_context))
{
    if (num_threads < 1)
        throw std::invalid_argument("at least one thread is required");
    workers.reserve(num_threads);
    for (int i = 0; i < num_threads; i++)
    {
        workers.push_back(std::async(std::launch::async, [this] { run_io_context(io_context); }));
    }
}

thread_pool::thread_pool(int num_threads, const std::vector<int> &affinity)
    : work_guard(boost::asio::make_work_guard(io_context))
{
    if (num_threads < 1)
        throw std::invalid_argument("at least one thread is required");
    workers.reserve(num_threads);
    for (int i = 0; i < num_threads; i++)
    {
        if (affinity.empty())
            workers.push_back(std::async(std::launch::async, [this] { run_io_context(io_context); }));
        else
        {
            int core = affinity[i % affinity.size()];
            workers.push_back(std::async(std::launch::async, [this, core] {
                set_affinity(core);
                run_io_context(io_context);
            }));
        }
    }
}

void thread_pool::set_affinity(int core)
{
#if SPEAD2_USE_PTHREAD_SETAFFINITY_NP
    if (core < 0 || core >= CPU_SETSIZE)
        log_warning("Core ID %1% is out of range for a CPU_SET", core);
    else
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core, &set);
        int status = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        if (status != 0)
        {
            std::error_code code(status, std::system_category());
            log_warning("Failed to bind to core %1%: %2% (%3%)", core, code.value(), code.message());
        }
    }
#else
    log_warning("Could not set affinity to core %1%: pthread_setaffinity_np not detected", core);
#endif
}

void thread_pool::stop()
{
    io_context.stop();
    for (auto &worker : workers)
    {
        try
        {
            worker.get();
        }
        catch (std::exception &e)
        {
            log_warning("worker thread threw an exception: %s", e.what());
        }
    }
    workers.clear();
}

thread_pool::~thread_pool()
{
    stop();
}


io_context_ref::io_context_ref(boost::asio::io_context &io_context)
    : io_context(io_context)
{
}

io_context_ref::io_context_ref(thread_pool &tpool)
    : io_context(tpool.get_io_context())
{
}

void io_context_ref::check_non_null(thread_pool *ptr)
{
    if (ptr == nullptr)
        throw std::invalid_argument("io_context_ref cannot be constructed from a null thread pool");
}

boost::asio::io_context &io_context_ref::operator*() const
{
    return io_context;
}

boost::asio::io_context *io_context_ref::operator->() const
{
    return &io_context;
}

std::shared_ptr<thread_pool> io_context_ref::get_shared_thread_pool() const &
{
    return thread_pool_holder;
}

std::shared_ptr<thread_pool> &&io_context_ref::get_shared_thread_pool() &&
{
    return std::move(thread_pool_holder);
}

} // namespace spead2
