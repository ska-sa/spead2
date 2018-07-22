/* Copyright 2018 SKA South Africa
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

#ifndef SPEAD2_COMMON_UNBOUNDED_QUEUE_H
#define SPEAD2_COMMON_UNBOUNDED_QUEUE_H

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <utility>
#include <mutex>
#include <queue>
#include <spead2/common_semaphore.h>
#include <spead2/common_ringbuffer.h>

namespace spead2
{

/**
 * A thread-safe unbounded queue.
 *
 * This behaves similarly to @ref ringbuffer, but it uses an unbounded queue
 * rather than a circular buffer, so pushes never block. It is less efficient
 * than a ringbuffer, and mainly intended for testing.
 *
 * Refer to the documentation on @ref ringbuffer for details of the design.
 */
template<typename T, typename DataSemaphore = semaphore>
class unbounded_queue
{
private:
    semaphore_fd data_sem;
    std::mutex mutex;
    bool stopped = false;
    std::queue<T> data;

    /**
     * Pop an item. The caller is responsible for semaphores and mutexes,
     * and there must be data available.
     */
    T pop_internal();

public:
    /**
     * Append an item to the queue. It uses move semantics, so on success, the
     * original value is undefined.
     *
     * @param value    Value to move
     * @throw ringbuffer_stopped if @ref stop is called first
     */
    void push(T &&value);

    /**
     * Construct a new item in the queue.
     *
     * @param args     Arguments to the constructor
     * @throw ringbuffer_stopped if @ref stop is called first
     */
    template<typename... Args>
    void emplace(Args&&... args);

    /**
     * Retrieve an item from the queue, if there is one.
     *
     * @throw ringbuffer_stopped if the queue is empty and @ref stop was called
     * @throw ringbuffer_empty if the queue is empty but still active
     */
    T try_pop();

    /**
     * Retrieve an item from the queue, blocking until there is one or until
     * the queue is stopped.
     *
     * @throw ringbuffer_stopped if the queue is empty and @ref stop was called
     */
    T pop();

    /**
     * Indicate that no more items will be produced. This does not immediately
     * stop consumers if there are still items in the queue; instead,
     * consumers will continue to retrieve remaining items, and will only be
     * signalled once the queue has drained.
     */
    void stop();

    /// Get access to the data semaphore
    const DataSemaphore &get_data_sem() const { return data_sem; }
};

template<typename T, typename DataSemaphore>
void unbounded_queue<T, DataSemaphore>::push(T &&value)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (stopped)
        throw ringbuffer_stopped();
    data.push(std::move(value));
    data_sem.put();
}

template<typename T, typename DataSemaphore>
template<typename... Args>
void unbounded_queue<T, DataSemaphore>::emplace(Args&&... args)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (stopped)
        throw ringbuffer_stopped();
    data.emplace(std::forward<Args>(args)...);
    data_sem.put();
}

template<typename T, typename DataSemaphore>
T unbounded_queue<T, DataSemaphore>::pop_internal()
{
    T result = std::move(data.front());
    data.pop();
    return result;
}

template<typename T, typename DataSemaphore>
T unbounded_queue<T, DataSemaphore>::try_pop()
{
    while (true)
    {
        int status = data_sem.try_get();
        std::lock_guard<std::mutex> lock(mutex);
        if (status == 0)
        {
            if (data.empty())
            {
                assert(stopped);  // only other reason we could have the got the semaphore
                data_sem.put();   // we didn't actually consume anything
                throw ringbuffer_stopped();
            }
            else
                return pop_internal();
        }
        else if (data.empty())
        {
            if (stopped)
                throw ringbuffer_stopped();
            else
                throw ringbuffer_empty();
        }
        /* We get here if data_sem.try_get was interrupted by a system call but
         * there is still data. Go around and try again.
         */
    }
}

template<typename T, typename DataSemaphore>
T unbounded_queue<T, DataSemaphore>::pop()
{
    semaphore_get(data_sem);
    std::lock_guard<std::mutex> lock(mutex);
    if (data.empty())
    {
        assert(stopped);
        data_sem.put();   // we didn't actually consume anything
        throw ringbuffer_stopped();
    }
    else
        return pop_internal();
}

template<typename T, typename DataSemaphore>
void unbounded_queue<T, DataSemaphore>::stop()
{
    std::lock_guard<std::mutex> lock(mutex);
    if (!stopped)
    {
        stopped = true;
        data_sem.put();  // wakes up waiters
    }
}

} // namespace spead2

#endif // SPEAD2_COMMON_UNBOUNDED_QUEUE_H
