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
 * Ring buffer.
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
#include <climits>
#include <iostream>
#include <spead2/common_logging.h>
#include <spead2/common_semaphore.h>

namespace spead2
{

/// Thrown when attempting to do a non-blocking push to a full ringbuffer
class ringbuffer_full : public std::runtime_error
{
public:
    ringbuffer_full() : std::runtime_error("ring buffer is full") {}
};

/// Thrown when attempting to do a non-blocking pop from an empty ringbuffer
class ringbuffer_empty : public std::runtime_error
{
public:
    ringbuffer_empty() : std::runtime_error("ring buffer is empty") {}
};

/**
 * Thrown when attempting to do a pop from an empty ringbuffer that has been
 * stopped, or a push to a ringbuffer that has been stopped.
 */
class ringbuffer_stopped : public std::runtime_error
{
public:
    ringbuffer_stopped() : std::runtime_error("ring buffer has been stopped") {}
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
    std::size_t head = 0;   ///< first slot with data
    bool stopped = false;   ///< Whether stop has been called

    /// Mutex held when writing to the tail (needed for safe multi-producer)
    std::mutex tail_mutex;
    std::size_t tail = 0;   ///< first slot without data
    /**
     * Position in the queue which the receiver should treat as "please stop",
     * or an invalid position if not yet stopped. Unlike @ref stopped, this is
     * updated with @ref tail_mutex rather than @ref head_mutex held. Once
     * set, it is guaranteed to always equal @ref tail.
     */
    std::size_t stop_position = SIZE_MAX;

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
    ~ringbuffer_base();

    /**
     * Check whether we're stopped and throw an appropriate error.
     *
     * @throw ringbuffer_stopped if the ringbuffer is empty and stopped
     * @throw ringbuffer_empty otherwise
     */
    void throw_empty_or_stopped();

    /**
     * Check whether we're stopped and throw an appropriate error.
     *
     * @throw ringbuffer_stopped if the ringbuffer is stopped
     * @throw ringbuffer_full otherwise
     */
    void throw_full_or_stopped();

    /// Implementation of pushing functions, which doesn't touch semaphores
    template<typename... Args>
    void emplace_internal(Args&&... args);

    /// Implementation of popping functions, which doesn't touch semaphores
    T pop_internal();

    /// Implementation of stopping, without the semaphores
    void stop_internal();
};

template<typename T>
T *ringbuffer_base<T>::get(std::size_t idx)
{
    return reinterpret_cast<T*>(&storage[idx]);
}

template<typename T>
void ringbuffer_base<T>::throw_empty_or_stopped()
{
    std::lock_guard<std::mutex> lock(head_mutex);
    if (head == stop_position)
        throw ringbuffer_stopped();
    else
        throw ringbuffer_empty();
}

template<typename T>
void ringbuffer_base<T>::throw_full_or_stopped()
{
    std::lock_guard<std::mutex> lock(tail_mutex);
    if (stopped)
        throw ringbuffer_stopped();
    else
        throw ringbuffer_full();
}

template<typename T>
ringbuffer_base<T>::ringbuffer_base(size_t cap)
    : storage(new storage_type[cap + 1]), cap(cap + 1), stop_position(cap + 1)
{
    /* We allocate one extra slot so that the destructor can disambiguate empty
     * from full. We could also use the semaphore values, but after stopping
     * they get used for other things so it is simplest not to rely on them.
     */
    assert(cap > 0);
}

template<typename T>
ringbuffer_base<T>::~ringbuffer_base()
{
    // Drain any remaining elements
    while (head != tail)
    {
        get(head)->~T();
        head = next(head);
    }
}

template<typename T>
template<typename... Args>
void ringbuffer_base<T>::emplace_internal(Args&&... args)
{
    std::lock_guard<std::mutex> lock(tail_mutex);
    if (stopped)
    {
        throw ringbuffer_stopped();
    }
    // Construct in-place
    new (get(tail)) T(std::forward<Args>(args)...);
    tail = next(tail);
}

template<typename T>
T ringbuffer_base<T>::pop_internal()
{
    std::lock_guard<std::mutex> lock(head_mutex);
    if (stop_position == head)
    {
        throw ringbuffer_stopped();
    }
    T result = std::move(*get(head));
    get(head)->~T();
    head = next(head);
    return result;
}

template<typename T>
void ringbuffer_base<T>::stop_internal()
{
    std::size_t saved_tail;

    {
        std::lock_guard<std::mutex> tail_lock(this->tail_mutex);
        if (stopped)
            return;
        stopped = true;
        saved_tail = tail;
    }

    {
        std::lock_guard<std::mutex> head_lock(this->head_mutex);
        stop_position = saved_tail;
    }
}

///////////////////////////////////////////////////////////////////////

/**
 * Ring buffer with blocking and non-blocking push and pop. It supports
 * non-copyable objects using move semantics. The producer may signal that it
 * has finished producing data by calling @ref stop, which will gracefully shut
 * down consumers as well as other producers. This class is fully thread-safe
 * for multiple producers and consumers.
 *
 * \internal
 *
 * The design is mostly standard: head and tail pointers, and semaphores
 * indicating the number of free and filled slots. One slot is always left
 * empty so that the destructor can distinguish between empty and full without
 * consulting the semaphores.
 *
 * The interesting part is @ref stop. On the producer side, this sets @ref
 * stopped, which is protected by the tail mutex to immediately prevent any
 * other pushes. On the consumer side, it sets @ref stop_position (protected
 * by the head mutex), so that consumers only get @ref ringbuffer_stopped
 * after consuming already-present elements. To wake everything up and prevent
 * any further waits, @ref stop ups both semaphores, and any functions that
 * observe a stop condition after downing a semaphore will re-up it. This
 * causes the semaphore to be transiently unavailable, which leads to the need
 * for @ref throw_empty_or_stopped and @ref throw_full_or_stopped.
 */
template<typename T, typename DataSemaphore = semaphore, typename SpaceSemaphore = semaphore>
class ringbuffer : protected ringbuffer_base<T>
{
private:
    DataSemaphore data_sem;     ///< Number of filled slots
    SpaceSemaphore space_sem;   ///< Number of available slots

public:
    explicit ringbuffer(std::size_t cap);

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
template<typename... Args>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::emplace(Args&&... args)
{
    semaphore_get(space_sem);
    try
    {
        this->emplace_internal(std::forward<Args...>(args...));
        data_sem.put();
    }
    catch (ringbuffer_stopped &e)
    {
        // We didn't actually use the slot we reserved with space_sem
        space_sem.put();
        throw;
    }
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
template<typename... Args>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::try_emplace(Args&&... args)
{
    if (space_sem.try_get() == -1)
        this->throw_full_or_stopped();
    try
    {
        this->emplace_internal(std::forward<Args...>(args...));
        data_sem.put();
    }
    catch (ringbuffer_stopped &e)
    {
        // We didn't actually use the slot we reserved with space_sem
        space_sem.put();
        throw;
    }
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::push(T &&value)
{
    semaphore_get(space_sem);
    try
    {
        this->emplace_internal(std::move(value));
        data_sem.put();
    }
    catch (ringbuffer_stopped &e)
    {
        // We didn't actually use the slot we reserved with space_sem
        space_sem.put();
        throw;
    }
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::try_push(T &&value)
{
    if (space_sem.try_get() == -1)
        this->throw_full_or_stopped();
    try
    {
        this->emplace_internal(std::move(value));
        data_sem.put();
    }
    catch (ringbuffer_stopped &e)
    {
        // We didn't actually use the slot we reserved with space_sem
        space_sem.put();
        throw;
    }
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
T ringbuffer<T, DataSemaphore, SpaceSemaphore>::pop()
{
    semaphore_get(data_sem);
    try
    {
        T result = this->pop_internal();
        space_sem.put();
        return result;
    }
    catch (ringbuffer_stopped &e)
    {
        // We didn't consume any data, wake up the next waiter
        data_sem.put();
        throw;
    }
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
T ringbuffer<T, DataSemaphore, SpaceSemaphore>::try_pop()
{
    if (data_sem.try_get() == -1)
        this->throw_empty_or_stopped();
    try
    {
        T result = this->pop_internal();
        space_sem.put();
        return result;
    }
    catch (ringbuffer_stopped &e)
    {
        // We didn't consume any data, wake up the next waiter
        data_sem.put();
        throw;
    }
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::stop()
{
    this->stop_internal();
    space_sem.put();
    data_sem.put();
}

} // namespace spead2

#endif // SPEAD2_COMMON_RINGBUFFER_H
