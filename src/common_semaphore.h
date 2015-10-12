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
 * Lightweight semaphore that does not support select()-like calls, and
 * which avoids kernel calls in the uncontended case.
 */
class semaphore
{
private:
    sem_t sem;

    /* This will match the value of the semaphore, except that it will be
     * transiently greater than it (since there is no way to atomically affect
     * both).
     *
     * However, once @ref stop is called, it will be one less than the
     * semaphore value. Thus, a reader that sees it negative knows that a stop
     * has arrived.
     */
    std::atomic<int> value;

    /// Implementation of @ref and @ref try_get once the semaphore is acquired
    int get_bottom();

public:
    explicit semaphore(int initial = 0);
    ~semaphore();

    /// Increment
    void put();

    /**
     * Decrement semaphore, blocking if necessary.
     *
     * @retval -1 if a system call was interrupted
     * @retval 0 if stop was called and the value is zero
     * @retval 1 on success
     */
    int get();

    /**
     * Decrement semaphore if possible, but do not block.
     *
     * @retval -1 if the semaphore is already zero or a system call was interrupted
     * @retval 0 if the semaphore is already zero and stopped
     * @retval 1 on success
     */
    int try_get();

    /**
     * Cause @ref get and @ref try_get to return 0 once the semaphore reaches
     * zero. It is not safe to call @ref post at the same time or after this is
     * called, nor is it safe to call @ref stop more than once.
     */
    void stop();
};

/////////////////////////////////////////////////////////////////////////////

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
    explicit semaphore_fd(int initial = 0);
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

    /// @copydoc semaphore::stop
    void stop();

    /// Return a file descriptor that will be readable when get will not block
    int get_fd() const;
};

/// Gets a semaphore, restarting automatically on interruptions
template<typename Semaphore>
static int semaphore_get(Semaphore &sem)
{
    while (true)
    {
        int result = sem.get();
        if (result != -1)
            return result;
    }
}

} // namespace spead2

#endif // SPEAD2_COMMON_SEMAPHORE_H
