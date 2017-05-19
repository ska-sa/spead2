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

#ifndef SPEAD2_COMMON_THREAD_POOL_H
#define SPEAD2_COMMON_THREAD_POOL_H

#include <type_traits>
#include <future>
#include <vector>
#include <array>
#include <cstdint>
#include <memory>
#include <boost/asio.hpp>
#include <spead2/recv_reader.h>

namespace spead2
{

/**
 * Combination of a @c boost::asio::io_service with a set of threads to handle
 * the callbacks. The threads are created by the constructor and shut down
 * and joined in the destructor.
 */
class thread_pool
{
private:
    boost::asio::io_service io_service;
    /// Prevents the io_service terminating automatically
    boost::asio::io_service::work work;
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

    /// Retrieve the embedded io_service
    boost::asio::io_service &get_io_service() { return io_service; }

    /// Shut down the thread pool
    void stop();
    /**
     * Set CPU affinity of current thread.
     */
    static void set_affinity(int core);
};

/**
 * A helper class that holds a reference to a @c boost::asio::io_service, and
 * optionally a shared pointer to a thread_pool. It is normally not explicitly
 * constructed, but other classes that need an @c io_service take it as an
 * argument and store it so that they can accept any of @c io_service, @c
 * thread_pool or @c std::shared_ptr<thread_pool>, and in the last case they
 * hold on to the reference.
 */
class io_service_ref
{
private:
    std::shared_ptr<thread_pool> thread_pool_holder;
    boost::asio::io_service &io_service;

    static void check_non_null(thread_pool *ptr);

public:
    /// Construct from a reference to an @c io_service
    io_service_ref(boost::asio::io_service &);
    /// Construct from a reference to a @ref thread_pool
    io_service_ref(thread_pool &);
    /**
     * Construct from a shared pointer to a @ref thread_pool. This is templated
     * so that it will also accept a shared pointer to a subclass of @ref
     * thread_pool.
     */
    template<typename T, typename SFINAE = typename std::enable_if<std::is_convertible<T *, thread_pool *>::value>::type>
    io_service_ref(std::shared_ptr<T>);

    /// Return the referenced @c io_service.
    boost::asio::io_service &operator*() const;
    /// Return a pointer to the referenced @c io_service.
    boost::asio::io_service *operator->() const;
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
io_service_ref::io_service_ref(std::shared_ptr<T> tpool)
    : thread_pool_holder((check_non_null(tpool.get()), std::move(tpool))),
    io_service(thread_pool_holder->get_io_service())
{
}

} // namespace spead2

#endif // SPEAD2_COMMON_THREAD_POOL_H
