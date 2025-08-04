/* Copyright 2015, 2021, 2023 National Research Foundation (SARAO)
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
#include <optional>
#include <cassert>
#include <climits>
#include <iostream>
#include <new>
#include <spead2/common_logging.h>
#include <spead2/common_semaphore.h>
#include <spead2/common_storage.h>

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

namespace detail
{

/// Sentinel counterpart to @ref ringbuffer_iterator
class ringbuffer_sentinel {};

/**
 * Basic iterator for @ref ringbuffer as well as ringbuffer-like classes: they
 * must provide @c pop which either returns when data is available or throws
 * @ref ringbuffer_stopped.
 *
 * This does not fully implement the iterator concept; it is suitable only for
 * range-based for loops.
 */
template<typename Ringbuffer>
class ringbuffer_iterator
{
private:
    using value_type = decltype(std::declval<Ringbuffer>().pop());

    Ringbuffer &owner;
    std::optional<value_type> current; // nullopt once ring has stopped

public:
    explicit ringbuffer_iterator(Ringbuffer &owner);
    bool operator!=(const ringbuffer_sentinel &);
    ringbuffer_iterator &operator++();
    value_type &&operator*();
};

template<typename Ringbuffer>
ringbuffer_iterator<Ringbuffer>::ringbuffer_iterator(Ringbuffer &owner)
    : owner(owner)
{
    ++*this;  // Load the first value
}

template<typename Ringbuffer>
bool ringbuffer_iterator<Ringbuffer>::operator!=(const ringbuffer_sentinel &)
{
    return bool(current);
}

template<typename Ringbuffer>
auto ringbuffer_iterator<Ringbuffer>::operator++() -> ringbuffer_iterator &
{
    /* Clear it first, so that we can reclaim the memory before making
     * space available in the ringbuffer, which might cause another
     * thread to allocate more memory.
     */
    current = std::nullopt;
    try
    {
        current = owner.pop();
    }
    catch (ringbuffer_stopped &)
    {
    }
    return *this;
}

template<typename Ringbuffer>
auto ringbuffer_iterator<Ringbuffer>::operator*() -> value_type &&
{
    return std::move(*current);
}

} // namespace detail

/**
 * Internal base class for @ref ringbuffer that is independent of the semaphore
 * type.
 */
template<typename T>
class ringbuffer_base
{
private:
    typedef detail::storage<T> storage_type;
    std::unique_ptr<storage_type[]> storage;
    const std::size_t cap;  ///< Number of slots

    /// Mutex held when reading from the head (needed for safe multi-consumer)
    mutable std::mutex head_mutex;
    std::size_t head = 0;   ///< first slot with data
    /**
     * Position in the queue which the receiver should treat as "please stop",
     * or an invalid position if not yet stopped. Unlike @ref stopped, this is
     * updated with @ref head_mutex rather than @ref tail_mutex held. Once
     * set, it is guaranteed to always equal @ref tail.
     */
    std::size_t stop_position = SIZE_MAX;

    /// Mutex held when writing to the tail (needed for safe multi-producer)
    mutable std::mutex tail_mutex;
    std::size_t tail = 0;   ///< first slot without data
    bool stopped = false;   ///< Whether stop has been called
    std::size_t producers = 0;  ///< Number of producers registered with @ref add_producer

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

    /// Discard the oldest item
    void discard_oldest_internal();

    /**
     * Implementation of stopping, without the semaphores.
     *
     * @param remove_producer If true, remove a producer and only stop if none are left
     * @returns whether the ringbuffer was stopped (i.e., this is the first call)
     */
    bool stop_internal(bool remove_producer = false);

public:
    /// Maximum number of items that can be held at once
    std::size_t capacity() const;

    /**
     * Return the number of items currently in the ringbuffer.
     *
     * This should only be used for metrics, not for control flow, as
     * the result could be out of date by the time it is returned.
     */
    std::size_t size() const;

    /**
     * Register a new producer. Producers only need to call this if they
     * want to call @ref ringbuffer::remove_producer.
     */
    void add_producer();
};

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
std::size_t ringbuffer_base<T>::capacity() const
{
    return cap - 1;
}

template<typename T>
std::size_t ringbuffer_base<T>::size() const
{
    std::lock_guard<std::mutex> head_lock(head_mutex);
    std::lock_guard<std::mutex> tail_lock(tail_mutex);
    if (head <= tail)
        return tail - head;
    else
        return tail + cap - head;
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
        storage[head].destroy();
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
    storage[tail].construct(std::forward<Args>(args)...);
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
    auto &item = storage[head];
    T result = std::move(*item);
    item.destroy();
    head = next(head);
    return result;
}

template<typename T>
void ringbuffer_base<T>::discard_oldest_internal()
{
    std::lock_guard<std::mutex> lock(head_mutex);
    if (stop_position < cap)
    {
        throw ringbuffer_stopped();
    }
    storage[head].destroy();
    head = next(head);
}

template<typename T>
bool ringbuffer_base<T>::stop_internal(bool remove_producer)
{
    std::size_t saved_tail;

    {
        std::lock_guard<std::mutex> tail_lock(this->tail_mutex);
        if (remove_producer)
        {
            assert(producers != 0);
            producers--;
            if (producers != 0)
                return false;
        }
        stopped = true;
        saved_tail = tail;
    }

    {
        std::lock_guard<std::mutex> head_lock(this->head_mutex);
        stop_position = saved_tail;
    }
    return true;
}

template<typename T>
void ringbuffer_base<T>::add_producer()
{
    std::lock_guard<std::mutex> tail_lock(this->tail_mutex);
    producers++;
}

///////////////////////////////////////////////////////////////////////

/**
 * Ring buffer with blocking and non-blocking push and pop. It supports
 * non-copyable objects using move semantics. The producer may signal that it
 * has finished producing data by calling @ref stop, which will gracefully shut
 * down consumers as well as other producers. This class is fully thread-safe
 * for multiple producers and consumers.
 *
 * With multiple producers it is sometimes desirable to only stop the
 * ringbuffer once all the producers are finished. To support this, a
 * producer may register itself with @ref add_producer, and indicate
 * completion with @ref remove_producer. If this causes the number of
 * producers to fall to zero, the stream is stopped.
 *
 * Normally, trying to push data when the ringbuffer is full will either block
 * (for @ref push and @ref emplace) or fail (for @ref try_push and
 * @ref try_emplace). However, if the ringbuffer is constructed with
 * @c discard_oldest set to true, then pushing to a full ringbuffer will
 * always succeed, dropping the oldest item if necessary. This can be useful
 * for lossy network applications where it is more important to have fresh
 * data.
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
 *
 * Another unusul
 */
template<typename T, typename DataSemaphore = semaphore, typename SpaceSemaphore = semaphore>
class ringbuffer : public ringbuffer_base<T>
{
private:
    const bool discard_oldest;
    DataSemaphore data_sem;     ///< Number of filled slots
    SpaceSemaphore space_sem;   ///< Number of available slots

    /// Implement @ref emplace and @ref try_emplace when discard_oldest is true
    template<typename... Args>
    void emplace_discard_oldest(Args&&... args);

public:
    explicit ringbuffer(std::size_t cap, bool discard_oldest = false);

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
     * @param value     Value to move
     * @param sem_args  Arbitrary arguments to pass to the space semaphore
     * @throw ringbuffer_stopped if @ref stop is called first
     */
    template<typename... SemArgs>
    void push(T &&value, SemArgs&&... sem_args);

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
     * @param sem_args  Arbitrary arguments to pass to the data semaphore
     * @throw ringbuffer_stopped if the queue is empty and @ref stop was called
     */
    template<typename... SemArgs>
    T pop(SemArgs&&... sem_args);

    /**
     * Indicate that no more items will be produced. This does not immediately
     * stop consumers if there are still items in the queue; instead,
     * consumers will continue to retrieve remaining items, and will only be
     * signalled once the queue has drained.
     *
     * @returns whether the ringbuffer was stopped
     */
    bool stop();

    /**
     * Indicate that a producer registered with @ref add_producer is
     * finished with the ringbuffer. If this was the last producer, the
     * ringbuffer is stopped.
     *
     * @returns whether the ringbuffer was stopped
     */
    bool remove_producer();

    /// Get access to the data semaphore
    const DataSemaphore &get_data_sem() const { return data_sem; }
    /// Get access to the free-space semaphore
    const SpaceSemaphore &get_space_sem() const { return space_sem; }
    /// Get the discard_oldest flag
    bool get_discard_oldest() const { return discard_oldest; }

    /**
     * Begin iteration over the items in the ringbuffer. This does not
     * return a full-blown iterator; it is only intended to be used for
     * a range-based for loop. For example:
     * <code>for (auto &&item : ringbuffer) { ... }</code>
     */
    detail::ringbuffer_iterator<ringbuffer> begin();
    /**
     * End iterator (see @ref begin).
     */
    detail::ringbuffer_sentinel end();
};

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
ringbuffer<T, DataSemaphore, SpaceSemaphore>::ringbuffer(size_t cap, bool discard_oldest)
    : ringbuffer_base<T>(cap), discard_oldest(discard_oldest), data_sem(0), space_sem(cap)
{
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
template<typename... Args>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::emplace_discard_oldest(Args&&... args)
{
    while (space_sem.try_get() == -1)
    {
        if (data_sem.try_get() == -1)
        {
            /* This could happen because consumers have removed all the data
             * before we could (in which case space_sem will be available in
             * the near future) or because we're contending with other
             * producers that have taken space_sem but not yet written the data
             * (in which case data_sem will be available in the near future).
             * Spin until we can get one of them.
             *
             * Spinning is not ideal, but I don't see any other way to wait
             * until one of the two semaphores has a token.
             */
            std::this_thread::yield();
            continue;
        }
        try
        {
            this->discard_oldest_internal();
        }
        catch (ringbuffer_stopped &)
        {
            data_sem.put();  // We didn't actually add any data
            throw;
        }
        /* discard_oldest_internal freed up space, which we're immediately
         * claiming, so we don't need to manipulate space_sem further.
         */
        break;
    }
    try
    {
        this->emplace_internal(std::forward<Args>(args)...);
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
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::emplace(Args&&... args)
{
    if (discard_oldest)
        return emplace_discard_oldest(std::forward<Args>(args)...);
    semaphore_get(space_sem);
    try
    {
        this->emplace_internal(std::forward<Args>(args)...);
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
    if (discard_oldest)
        return emplace_discard_oldest(std::forward<Args>(args)...);
    /* TODO: try_get needs to be modified to distinguish between interrupted
     * system calls and zero semaphore (EAGAIN vs EINTR). But in most (all?)
     * cases is impossible because try_get is not blocking.
     */
    if (space_sem.try_get() == -1)
        this->throw_full_or_stopped();
    try
    {
        this->emplace_internal(std::forward<Args>(args)...);
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
template<typename... SemArgs>
void ringbuffer<T, DataSemaphore, SpaceSemaphore>::push(T &&value, SemArgs&&... sem_args)
{
    if (discard_oldest)
        return emplace_discard_oldest(std::move(value));
    semaphore_get(space_sem, std::forward<SemArgs>(sem_args)...);
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
    if (discard_oldest)
        return emplace_discard_oldest(std::move(value));
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
template<typename... SemArgs>
T ringbuffer<T, DataSemaphore, SpaceSemaphore>::pop(SemArgs&&... sem_args)
{
    semaphore_get(data_sem, std::forward<SemArgs>(sem_args)...);
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
bool ringbuffer<T, DataSemaphore, SpaceSemaphore>::stop()
{
    if (this->stop_internal())
    {
        space_sem.put();
        data_sem.put();
        return true;
    }
    else
        return false;
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
bool ringbuffer<T, DataSemaphore, SpaceSemaphore>::remove_producer()
{
    if (this->stop_internal(true))
    {
        space_sem.put();
        data_sem.put();
        return true;
    }
    else
        return false;
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
auto ringbuffer<T, DataSemaphore, SpaceSemaphore>::begin() -> detail::ringbuffer_iterator<ringbuffer>
{
    return detail::ringbuffer_iterator(*this);
}

template<typename T, typename DataSemaphore, typename SpaceSemaphore>
detail::ringbuffer_sentinel ringbuffer<T, DataSemaphore, SpaceSemaphore>::end()
{
    return detail::ringbuffer_sentinel();
}

} // namespace spead2

#endif // SPEAD2_COMMON_RINGBUFFER_H
