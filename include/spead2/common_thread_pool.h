/* Copyright 2015, 2023, 2025 National Research Foundation (SARAO)
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

#ifndef SPEAD2_COMMON_THREAD_POOL_H
#define SPEAD2_COMMON_THREAD_POOL_H

#include <type_traits>
#include <future>
#include <vector>
#include <array>
#include <cstdint>
#include <memory>
#include <boost/asio.hpp>

namespace spead2
{

/**
 * Combination of a @c boost::asio::io_context with a set of threads to handle
 * the callbacks. The threads are created by the constructor and shut down
 * and joined in the destructor.
 */
class thread_pool
{
private:
    boost::asio::io_context io_context;
    /// Prevents the io_context terminating automatically
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work_guard;
    /**
     * Futures that becomes ready when a worker thread completes. It
     * is connected to an async task.
     */
    std::vector<std::future<void> > workers;

public:
    explicit thread_pool(int num_threads = 1);
    /**
     * Construct with explicit core affinity for the threads. The @a affinity
     * list can be shorter or longer than @a num_threads. Threads are allocated
     * in round-robin fashion to cores. Failures to set affinity are logged
     * but do not cause an exception.
     */
    thread_pool(int num_threads, const std::vector<int> &affinity);
    ~thread_pool();

    /// Retrieve the embedded io_context
    boost::asio::io_context &get_io_context() { return io_context; }
    /// Retrieve the embedded io_context (deprecated alias)
    [[deprecated("use get_io_context")]]
    boost::asio::io_context &get_io_service() { return io_context; }

    /// Shut down the thread pool
    void stop();
    /**
     * Set CPU affinity of current thread.
     */
    static void set_affinity(int core);
};

/**
 * A helper class that holds a reference to a @c boost::asio::io_context, and
 * optionally a shared pointer to a thread_pool. It is normally not explicitly
 * constructed, but other classes that need an @c io_context take it as an
 * argument and store it so that they can accept any of @c io_context, @c
 * thread_pool or @c std::shared_ptr<thread_pool>, and in the last case they
 * hold on to the reference.
 */
class io_context_ref
{
private:
    std::shared_ptr<thread_pool> thread_pool_holder;
    boost::asio::io_context &io_context;

    static void check_non_null(thread_pool *ptr);

public:
    /// Construct from a reference to an @c io_context
    io_context_ref(boost::asio::io_context &);
    /// Construct from a reference to a @ref thread_pool
    io_context_ref(thread_pool &);
    /**
     * Construct from a shared pointer to a @ref thread_pool. This is templated
     * so that it will also accept a shared pointer to a subclass of @ref
     * thread_pool.
     */
    template<typename T, typename SFINAE = std::enable_if_t<std::is_convertible_v<T *, thread_pool *>>>
    io_context_ref(std::shared_ptr<T>);

    /// Return the referenced @c io_context.
    boost::asio::io_context &operator*() const;
    /// Return a pointer to the referenced @c io_context.
    boost::asio::io_context *operator->() const;
    /// Return the shared pointer to the @ref thread_pool, if constructed from one.
    std::shared_ptr<thread_pool> get_shared_thread_pool() const &;
    /**
     * Return the shared pointer to the @ref thread_pool, if constructed from
     * one. This overload returns an rvalue reference, allowing the shared
     * pointer to be moved out of a temporary.
     */
    std::shared_ptr<thread_pool> &&get_shared_thread_pool() &&;
};

template<typename T, typename SFINAE>
io_context_ref::io_context_ref(std::shared_ptr<T> tpool)
    : thread_pool_holder((check_non_null(tpool.get()), std::move(tpool))),
    io_context(thread_pool_holder->get_io_context())
{
}

using io_service_ref [[deprecated("use io_context_ref instead")]] = io_context_ref;

} // namespace spead2

#endif // SPEAD2_COMMON_THREAD_POOL_H
