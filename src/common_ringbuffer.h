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
 *
 * Various types of ring buffers.
 */

#ifndef SPEAD2_COMMON_RINGBUFFER_H
#define SPEAD2_COMMON_RINGBUFFER_H

#include <mutex>
#include <condition_variable>
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <cassert>
#include <iostream>
#include "common_logging.h"
#include "common_semaphore.h"

namespace spead2
{

/// Thrown when attempting to do a non-blocking push to a full ringbuffer
class ringbuffer_full : public std::runtime_error
{
public:
    ringbuffer_full() : std::runtime_error("ring buffer is full") {}
};

/// Thrown when attempting to do a non-blocking pop from a full ringbuffer
class ringbuffer_empty : public std::runtime_error
{
public:
    ringbuffer_empty() : std::runtime_error("ring buffer is empty") {}
};

/// Thrown when attempting to do a pop from an empty ringbuffer that has been shut down
class ringbuffer_stopped : public std::runtime_error
{
public:
    ringbuffer_stopped() : std::runtime_error("ring buffer has been shut down") {}
};

/**
 * Internal base class for @ref ringbuffer that is independent of the semaphore
 * type.
 */
template<typename T>
class ringbuffer_base
{
private:
    typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type storage_type;
    std::unique_ptr<storage_type[]> storage;
    const std::size_t cap;  ///< Number of slots

    /// Mutex held when reading from the head (needed for safe multi-consumer)
    std::mutex head_mutex;
    /// Mutex held when writing to the tail (needed for safe multi-producer)
    std::mutex tail_mutex;
    std::size_t head = 0;   ///< first slot with data
    std::size_t tail = 0;   ///< first slot without data

    /// Gets pointer to the slot number @a idx
    T *get(std::size_t idx);

    /// Increments @a idx, wrapping around.
    std::size_t next(std::size_t idx)
    {
        idx++;
        if (idx == cap)
            idx = 0;
        return idx;
    }

protected:
    explicit ringbuffer_base(std::size_t cap);

    /// Implementation of pushing functions, which doesn't touch semaphores
    template<typename... Args>
    void emplace_internal(Args&&... args);

    /// Implementation of popping functions, which doesn't touch semaphores
    T pop_internal();
};

template<typename T>
T *ringbuffer_base<T>::get(std::size_t idx)
{
    return reinterpret_cast<T*>(&storage[idx]);
}

template<typename T>
ringbuffer_base<T>::ringbuffer_base(size_t cap)
    : storage(new storage_type[cap]), cap(cap)
{
    assert(cap > 0);
}

template<typename T>
template<typename... Args>
void ringbuffer_base<T>::emplace_internal(Args&&... args)
{
    std::lock_guard<std::mutex> lock(tail_mutex);
    // Construct in-place
    new (get(tail)) T(std::forward<Args>(args)...);
    tail = next(tail);
}

template<typename T>
T ringbuffer_base<T>::pop_internal()
{
    std::lock_guard<std::mutex> lock(head_mutex);
    T result = std::move(*get(head));
    get(head)->~T();
    head = next(head);
    return result;
}

///////////////////////////////////////////////////////////////////////

/**
 * Ring buffer with blocking and non-blocking push and pop. It supports
 * non-copyable objects using move semantics. The producer may signal that it
 * has finished producing data by calling @ref stop, which will gracefully shut
 * down the consumer.
 */
template<typename T, typename DataSemaphore = semaphore, typename SpaceSemaphore = semaphore>
class ringbuffer : protected ringbuffer_base<T>
{
private:
    DataSemaphore data_sem;
    SpaceSemaphore space_sem;
    bool stopped = false;   ///< Protects @a stop from multiple calls

public:
    explicit ringbuffer(std::size_t cap);
    ~ringbuffer();

    /**
     * Append an item to the queue, if there is space. It uses move
     * semantics, so on success, the original value is undefined.
     *
     * @param value    Value to move
     * @throw ringbuffer_full if there is no space
     * @throw ringbuffer_stopped if @ref stop has already been called
     */
    void try_push(T &&value);

    /**
     * Construct a new item in the queue, if there is space.
     *
     * @param args     Arguments to the constructor
     * @throw ringbuffer_full if there is no space
     * @throw ringbuffer_stopped if @ref stop has already been called
     */
    template<typename... Args>
    void try_emplace(Args&&... args);

    /**
     * Append an item to the queue, blocking if necessary. It uses move
     * semantics, so on success, the original value is undefined.
     *
     * @param value    Value to move
     * @throw ringbuffer_stopped if @ref stop is called first
     */
    void push(T &&value);

    /**
     * Construct a new item in the queue, blocking if necessary.
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
     *
     * It is safe to call this function multiple times, but it is not
     * thread-safe.  It is not legal to call @ref push, @ref try_push, @ref
     * emplace or @ref try_emplace at the same time or after calling this
     * function.
     */
    void stop();

    /// Get access to the data semaphore
    const DataSemaphore &get_data_sem() const { return data_sem; }
    /// Get access to the free-space semaphore
    const SpaceSemaphore &get_space_sem() const { return space_sem; }
};

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
ringbuffer<T, DataSemaphore, SpaceSemaphore>::ringbuffer(size_t cap)
    : ringbuffer_base<T>(cap), data_sem(0), space_sem(cap)
{
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
ringbuffer<T, DataSemaphore, SpaceSemaphore>::~ringbuffer()
{
    // Drain any remaining elements
    while (data_sem.try_get() == 1)
    {
        this->pop_internal();
    }
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
template<typename... Args>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::emplace(Args&&... args)
{
    semaphore_get(space_sem);
    this->emplace_internal(std::forward<Args...>(args...));
    data_sem.put();
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
template<typename... Args>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::try_emplace(Args&&... args)
{
    if (space_sem.try_get() == 1)
    {
        this->emplace_internal(std::forward<Args...>(args...));
        data_sem.put();
    }
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::push(T &&value)
{
    semaphore_get(space_sem);
    this->emplace_internal(std::move(value));
    data_sem.put();
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::try_push(T &&value)
{
    if (space_sem.try_get() == 1)
    {
        this->emplace_internal(std::move(value));
        data_sem.put();
    }
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
T ringbuffer<T, DataSemaphore, SpaceSemaphore>::pop()
{
    int status = semaphore_get(data_sem);
    if (status == 0)
        throw ringbuffer_stopped();
    T result = this->pop_internal();
    space_sem.put();
    return result;
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
T ringbuffer<T, DataSemaphore, SpaceSemaphore>::try_pop()
{
    int status = data_sem.try_get();
    if (status == -1)
        throw ringbuffer_empty();
    else if (status == 0)
        throw ringbuffer_stopped();
    T result = this->pop_internal();
    space_sem.put();
    return result;
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::stop()
{
    if (!stopped)
    {
        stopped = true;
        data_sem.stop();
    }
}

} // namespace spead2

#endif // SPEAD2_COMMON_RINGBUFFER_H
