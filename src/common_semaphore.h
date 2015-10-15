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

#include <memory>
#include <atomic>
#include <semaphore.h>

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

    /// @copydoc semaphore::put
    void put();

    /// @copydoc semaphore::get
    int get();

    /// @copydoc semaphore::try_get
    int try_get();
};

/**
 * Semaphore that uses file descriptors, so that it can be plumbed
 * into an event loop.
 */
class semaphore_fd
{
private:
    int pipe_fds[2];

    // Prevent copying: semaphores are not copyable resources
    semaphore_fd(const semaphore_fd &) = delete;
    semaphore_fd &operator=(const semaphore_fd &) = delete;

public:
    explicit semaphore_fd(unsigned int initial = 0);
    ~semaphore_fd();

    /// Move constructor
    semaphore_fd(semaphore_fd &&);
    /// Move assignment
    semaphore_fd &operator=(semaphore_fd &&);

    /// @copydoc semaphore::put
    void put();

    /// @copydoc semaphore::get
    int get();

    /// @copydoc semaphore::try_get
    int try_get();

    /// Return a file descriptor that will be readable when get will not block
    int get_fd() const;
};

/////////////////////////////////////////////////////////////////////////////

#if !__APPLE__
/**
 * Lightweight semaphore that does not support select()-like calls, and
 * which avoids kernel calls in the uncontended case.
 */
class semaphore
{
private:
    sem_t sem;

public:
    explicit semaphore(unsigned int initial = 0);
    ~semaphore();

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

#else // __APPLE__

// OS X doesn't implement POSIX semaphores, so we fall back to the fd-based semaphores,
// but keep it a different type to avoid exposing get_fd.
class semaphore : private semaphore_fd
{
public:
    using semaphore_fd::semaphore_fd;
    using semaphore_fd::put;
    using semaphore_fd::get;
    using semaphore_fd::try_get;
};

#endif // __APPLE__

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
