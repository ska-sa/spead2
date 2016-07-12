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
 * Semaphore types. There are two types that share a common API, but not a
 * common base class.
 */

#ifndef SPEAD2_COMMON_SEMAPHORE_H
#define SPEAD2_COMMON_SEMAPHORE_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#include <memory>
#include <atomic>
#if SPEAD2_USE_POSIX_SEMAPHORES
# include <semaphore.h>
#endif

namespace spead2
{

/**
 * Semaphore that uses no system calls, instead spinning on an atomic.
 *
 * This is useful for a heavily contended ringbuffer with low capacity.
 * In most other cases, increasing the ringbuffer size is sufficient.
 */
class semaphore_spin
{
private:
    std::atomic<unsigned int> value;

public:
    explicit semaphore_spin(unsigned int initial = 0);

    /// Increment
    void put();

    /**
     * Decrement semaphore, blocking if necessary.
     *
     * @retval -1 if a system call was interrupted
     * @retval 0 on success
     */
    int get();

    /**
     * Decrement semaphore if possible, but do not block.
     *
     * @retval -1 if the semaphore is already zero or a system call was interrupted
     * @retval 0 on success
     */
    int try_get();
};

/**
 * Semaphore that uses file descriptors, so that it can be plumbed
 * into an event loop.
 */
class semaphore_pipe
{
private:
    int pipe_fds[2];

    // Prevent copying: semaphores are not copyable resources
    semaphore_pipe(const semaphore_pipe &) = delete;
    semaphore_pipe &operator=(const semaphore_pipe &) = delete;

public:
    explicit semaphore_pipe(unsigned int initial = 0);
    ~semaphore_pipe();

    /// Move constructor
    semaphore_pipe(semaphore_pipe &&);
    /// Move assignment
    semaphore_pipe &operator=(semaphore_pipe &&);

    /// @copydoc semaphore_spin::put
    void put();

    /// @copydoc semaphore_spin::get
    int get();

    /// @copydoc semaphore_spin::try_get
    int try_get();

    /// Return a file descriptor that will be readable when get will not block
    int get_fd() const;
};

/**
 * Variant of @ref semaphore_pipe that uses eventfd(2) instead of a pipe.
 */
class semaphore_eventfd
{
private:
    int fd;

    // Prevent copying: semaphores are not copyable resources
    semaphore_eventfd(const semaphore_eventfd &) = delete;
    semaphore_eventfd &operator=(const semaphore_eventfd &) = delete;

public:
    explicit semaphore_eventfd(unsigned int initial = 0);
    ~semaphore_eventfd();

    /// Move constructor
    semaphore_eventfd(semaphore_eventfd &&);
    /// Move assignment
    semaphore_eventfd &operator=(semaphore_eventfd &&);

    /// @copydoc semaphore_spin::put
    void put();

    /// @copydoc semaphore_spin::get
    int get();

    /// @copydoc semaphore_spin::try_get
    int try_get();

    /// @copydoc semaphore_pipe::get_fd
    int get_fd() const;
};


/////////////////////////////////////////////////////////////////////////////

#if SPEAD2_USE_POSIX_SEMAPHORES
/**
 * Lightweight semaphore that does not support select()-like calls, and
 * which avoids kernel calls in the uncontended case.
 */
class semaphore_posix
{
private:
    sem_t sem;

public:
    explicit semaphore_posix(unsigned int initial = 0);
    ~semaphore_posix();

    /// @copydoc semaphore_spin::put
    void put();

    /// @copydoc semaphore_spin::get
    int get();

    /// @copydoc semaphore_spin::try_get
    int try_get();
};

#endif // SPEAD2_USE_POSIX_SEMAPHORES

#if SPEAD2_USE_EVENTFD
typedef semaphore_eventfd semaphore_fd;
#else
typedef semaphore_pipe semaphore_fd;
#endif

#if SPEAD2_USE_POSIX_SEMAPHORES

typedef semaphore_posix semaphore;

#else

// Fall back to the file descriptor-based semaphores, but keep it a different
// type to avoid exposing get_fd.
class semaphore : private semaphore_fd
{
public:
    using semaphore_fd::semaphore_fd;
    using semaphore_fd::put;
    using semaphore_fd::get;
    using semaphore_fd::try_get;
};

#endif // !SPEAD2_USE_POSIX_SEMAPHORES

/////////////////////////////////////////////////////////////////////////////

/// Gets a semaphore, restarting automatically on interruptions
template<typename Semaphore>
static void semaphore_get(Semaphore &sem)
{
    while (sem.get() == -1)
    {
    }
}

} // namespace spead2

#endif // SPEAD2_COMMON_SEMAPHORE_H
