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
 * Semaphore that uses file descriptors, so that it can be plumbed
 * into an event loop.
 */

#ifndef SPEAD2_COMMON_SEMAPHORE_H
#define SPEAD2_COMMON_SEMAPHORE_H

namespace spead2
{

/**
 * Semaphore that uses file descriptors, so that it can be plumbed
 * into an event loop.
 */
class semaphore
{
private:
    int pipe_fds[2];

    // Prevent copying: semaphores are not copyable resources
    semaphore(const semaphore &) = delete;
    semaphore &operator=(const semaphore &) = delete;

public:
    /// Move constructor
    semaphore(semaphore &&);
    /// Move assignment
    semaphore &operator=(semaphore &&);

    /// Increment semaphore.
    void put();
    /**
     * Decrement semaphore, blocking if necessary, but aborting if the
     * system call was interrupted.
     *
     * @retval -1 if the wait was interrupted
     * @retval 0 if @ref stop was called
     * @retval 1 on success
     */
    int get();

    /// Interrupt any current waiters and release resources
    void stop();

    /// Return a file descriptor that will be readable when get will not block
    int get_fd() const;

    semaphore();
    ~semaphore();
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
