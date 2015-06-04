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
 * Ring buffer with blocking and non-blocking pop, but only non-blocking push.
 * It supports non-copyable objects using move semantics. The producer may
 * signal that it has finished producing data by calling @ref stop_unlocked,
 * which will gracefully shut down the consumer.
 *
 * This is a base class that cannot be used directly. Derived classes must
 * provide the synchronisation mechanisms.
 */
template<typename T>
class ringbuffer_base
{
private:
    typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type storage_type;

    std::unique_ptr<storage_type[]> storage;
    const std::size_t cap;  ///< Number of slots
    std::size_t head = 0;   ///< First slot with data
    std::size_t tail = 0;   ///< First free slot
    std::size_t len = 0;    ///< Number of slots with data
    bool stopped = false;

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
    /**
     * Checks whether the ringbuffer is empty.
     *
     * @throw ringbuffer_stopped if the ringbuffer is empty and has been stopped.
     * @pre The caller holds a mutex
     */
    bool empty_unlocked() const
    {
        if (len == 0 && stopped)
            throw ringbuffer_stopped();
        return len == 0;
    }

    bool is_stopped_unlocked() const
    {
        return stopped;
    }

    void stop_unlocked()
    {
        stopped = true;
    }

    /**
     * Construct a new item in the queue, if there is space.
     *
     * @param args     Arguments to the constructor
     * @throw ringbuffer_full if there is no space
     * @throw ringbuffer_stopped if @ref stop_unlocked has already been called
     *
     * @pre The caller holds a mutex
     */
    template<typename... Args>
    void try_emplace_unlocked(Args&&... args);

    /**
     * Pops an item from the ringbuffer and returns it.
     *
     * @pre The caller holds a mutex, and there is data available.
     */
    T pop_unlocked();

    explicit ringbuffer_base(std::size_t cap);
public:
    ~ringbuffer_base();
};

template<typename T>
ringbuffer_base<T>::ringbuffer_base(std::size_t cap)
    : storage(new storage_type[cap]), cap(cap)
{
    assert(cap > 0);
}

template<typename T>
ringbuffer_base<T>::~ringbuffer_base()
{
    // Drain any remaining elements
    while (len > 0)
    {
        get(head)->~T();
        head = next(head);
        len--;
    }
}

template<typename T>
T *ringbuffer_base<T>::get(std::size_t idx)
{
    return reinterpret_cast<T*>(&storage[idx]);
}

template<typename T>
template<typename... Args>
void ringbuffer_base<T>::try_emplace_unlocked(Args&&... args)
{
    if (stopped)
        throw ringbuffer_stopped();
    if (len == cap)
        throw ringbuffer_full();
    // Construct in-place
    new (get(tail)) T(std::forward<Args>(args)...);
    // Advance the queue
    tail = next(tail);
    len++;
}

template<typename T>
T ringbuffer_base<T>::pop_unlocked()
{
    T result = std::move(*get(head));
    get(head)->~T();
    head = next(head);
    len--;
    return result;
}

///////////////////////////////////////////////////////////////////////

/**
 * Implementation of @ref ringbuffer_base that uses condition variables
 * for inter-thread signalling.
 */
template<typename T>
class ringbuffer_cond : public ringbuffer_base<T>
{
protected:
    /// Protects access to the internal fields
    std::mutex mutex;

    /// Signalled when data is added or @a stop is called
    std::condition_variable data_cond;

public:
    /**
     * Constructs an empty ringbuffer.
     *
     * @param cap      Maximum capacity, in items
     */
    explicit ringbuffer_cond(std::size_t cap);

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
     * It is safe to call this function multiple times.
     */
    void stop();
};

template<typename T>
ringbuffer_cond<T>::ringbuffer_cond(std::size_t cap) : ringbuffer_base<T>(cap)
{
}

template<typename T>
void ringbuffer_cond<T>::try_push(T &&value)
{
    // Construct in-place with move constructor
    try_emplace(std::move(value));
}

template<typename T>
template<typename... Args>
void ringbuffer_cond<T>::try_emplace(Args&&... args)
{
    std::unique_lock<std::mutex> lock(mutex);
    this->try_emplace_unlocked(std::forward<Args>(args)...);
    data_cond.notify_one();
}

template<typename T>
T ringbuffer_cond<T>::try_pop()
{
    std::unique_lock<std::mutex> lock(mutex);
    if (this->empty_unlocked())
    {
        throw ringbuffer_empty();
    }
    return this->pop_unlocked();
}

template<typename T>
T ringbuffer_cond<T>::pop()
{
    std::unique_lock<std::mutex> lock(mutex);
    while (this->empty_unlocked())
    {
        data_cond.wait(lock);
    }
    return this->pop_unlocked();
}

template<typename T>
void ringbuffer_cond<T>::stop()
{
    std::unique_lock<std::mutex> lock(mutex);
    this->stop_unlocked();
    data_cond.notify_all();
}

///////////////////////////////////////////////////////////////////////

/**
 * Implementation of @ref ringbuffer_base that uses @ref semaphore or a
 * related class for inter-thread signalling. This is slower and can block the
 * sender if the number of entries exceeds the capacity of a pipe, but can be
 * used with poll()-style asynchronous operations.
 */
template<typename T, typename Semaphore = semaphore>
class ringbuffer_semaphore : public ringbuffer_base<T>
{
protected:
    /// Protects access to the internal fields
    std::mutex mutex;

    Semaphore sem;
public:
    /**
     * Constructs an empty ringbuffer.
     *
     * @param cap      Maximum capacity, in items
     */
    explicit ringbuffer_semaphore(std::size_t cap);

    ~ringbuffer_semaphore();

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
     * It is safe to call this function multiple times.
     */
    void stop();

    /**
     * Return a file descriptor to poll() on to get notification when pop()
     * can proceed without blocking, or try_pop will succeed. Do not read
     * or write anything from/to this descriptor.
     */
    int get_fd() const { return sem.get_fd(); }
};

template<typename T, typename Semaphore>
ringbuffer_semaphore<T, Semaphore>::ringbuffer_semaphore(std::size_t cap) : ringbuffer_base<T>(cap)
{
}

template<typename T, typename Semaphore>
ringbuffer_semaphore<T, Semaphore>::~ringbuffer_semaphore()
{
}

template<typename T, typename Semaphore>
void ringbuffer_semaphore<T, Semaphore>::try_push(T &&value)
{
    // Construct in-place with move constructor
    try_emplace(std::move(value));
}

template<typename T, typename Semaphore>
template<typename... Args>
void ringbuffer_semaphore<T, Semaphore>::try_emplace(Args&&... args)
{
    std::unique_lock<std::mutex> lock(mutex);
    this->try_emplace_unlocked(std::forward<Args>(args)...);
    sem.put();
}

template<typename T, typename Semaphore>
T ringbuffer_semaphore<T, Semaphore>::try_pop()
{
    std::unique_lock<std::mutex> lock(mutex);
    if (this->empty_unlocked())
    {
        throw ringbuffer_empty();
    }
    // This should not block, because there is an entry to consume
    int status = semaphore_get(sem);
    assert(status == 1); (void) status;
    return this->pop_unlocked();
}

template<typename T, typename Semaphore>
T ringbuffer_semaphore<T, Semaphore>::pop()
{
    int status = semaphore_get(sem);
    if (status == 0)
        throw ringbuffer_stopped();

    std::unique_lock<std::mutex> lock(mutex);
    assert(!this->empty_unlocked());
    return this->pop_unlocked();
}

template<typename T, typename Semaphore>
void ringbuffer_semaphore<T, Semaphore>::stop()
{
    std::unique_lock<std::mutex> lock(mutex);
    this->stop_unlocked();
    sem.stop();
}

} // namespace spead2

#endif // SPEAD2_COMMON_RINGBUFFER_H
