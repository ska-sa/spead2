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

#include <cerrno>
#include <system_error>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <atomic>
#include "common_semaphore.h"
#include "common_logging.h"

namespace spead2
{

[[noreturn]] static void throw_errno()
{
    throw std::system_error(errno, std::system_category());
}

static void log_errno(const char *format)
{
    std::error_code code(errno, std::system_category());
    log_warning(format, code.value(), code.message());
}

semaphore::semaphore(int initial) : value(initial)
{
    if (sem_init(&sem, 0, initial) == -1)
        throw_errno();
}

semaphore::~semaphore()
{
    if (sem_destroy(&sem) == -1)
    {
        // Destructor, so can't throw
        log_errno("failed to destroy semaphore: %1% (%2%)");
    }
}

void semaphore::put()
{
    value.fetch_add(1, std::memory_order_release);
    if (sem_post(&sem) == -1)
    {
        value--; // Undo the += above
        throw_errno();
    }
}

void semaphore::stop()
{
    // Wake up at least one waiter, which will wake up others
    if (sem_post(&sem) == -1)
        throw_errno();
}

int semaphore::get_bottom()
{
    int old = value.fetch_sub(1, std::memory_order_acquire);
    if (old <= 0)
    {
        // The new value is negative, which means we have a stop
        // rather than a real token. Wake up another waiter.
        value.fetch_add(1, std::memory_order_release);
        if (sem_post(&sem) == -1)
        {
            value--; // undo the +1
            throw_errno();
        }
        return 0;
    }
    else
        return 1;
}

int semaphore::try_get()
{
    int status = sem_trywait(&sem);
    if (status == -1)
    {
        if (errno == EAGAIN || errno == EINTR)
            return -1;
        else
            throw_errno();
    }
    else
        return get_bottom();
}

int semaphore::get()
{
    int status = sem_wait(&sem);
    if (status == -1)
    {
        if (errno == EINTR)
            return -1;
        else
            throw_errno();
    }
    else
        return get_bottom();
}

/////////////////////////////////////////////////////////////////////////////

semaphore_fd::semaphore_fd(semaphore_fd &&other)
{
    for (int i = 0; i < 2; i++)
    {
        pipe_fds[i] = other.pipe_fds[i];
        other.pipe_fds[i] = -1;
    }
}

semaphore_fd &semaphore_fd::operator=(semaphore_fd &&other)
{
    for (int i = 0; i < 2; i++)
    {
        if (pipe_fds[i] != -1)
        {
            if (close(pipe_fds[i]) == -1)
                throw_errno();
            pipe_fds[i] = -1;
        }
    }
    for (int i = 0; i < 2; i++)
    {
        pipe_fds[i] = other.pipe_fds[i];
        other.pipe_fds[i] = -1;
    }
    return *this;
}

semaphore_fd::semaphore_fd(int initial)
{
    if (pipe(pipe_fds) == -1)
        throw_errno();
    for (int i = 0; i < 2; i++)
    {
        int flags = fcntl(pipe_fds[i], F_GETFD);
        if (flags == -1)
            throw_errno();
        flags |= FD_CLOEXEC;
        if (fcntl(pipe_fds[i], F_SETFD, flags) == -1)
            throw_errno();
    }
    // Make the read end non-blocking, for try_get
    int flags = fcntl(pipe_fds[0], F_GETFL);
    if (flags == -1)
        throw_errno();
    flags |= O_NONBLOCK;
    if (fcntl(pipe_fds[0], F_SETFL, flags) == -1)
        throw_errno();
    // TODO: this could probably be optimised
    for (int i = 0; i < initial; i++)
        put();
}

semaphore_fd::~semaphore_fd()
{
    for (int i = 0; i < 2; i++)
        if (pipe_fds[i] != -1)
        {
            if (close(pipe_fds[i]) == -1)
            {
                // Can't throw, because this is a destructor
                log_errno("failed to close pipe: %1% (%2%)");
            }
        }
}

void semaphore_fd::put()
{
    char byte = 0;
    int status;
    do
    {
        status = write(pipe_fds[1], &byte, 1);
        if (status < 0 && errno != EINTR)
        {
            throw_errno();
        }
    } while (status < 0);
}

int semaphore_fd::get()
{
    char byte = 0;
    while (true)
    {
        struct pollfd pfd = {};
        pfd.fd = pipe_fds[0];
        pfd.events = POLLIN;
        int status = poll(&pfd, 1, -1);
        if (status == -1)
        {
            if (errno == EINTR)
                return -1;
            else
                throw_errno();
        }
        status = read(pipe_fds[0], &byte, 1);
        if (status >= 0)
            return status; // either success or write end closed
        else if (errno != EAGAIN && errno != EWOULDBLOCK)
            throw_errno();
    }
}

int semaphore_fd::try_get()
{
    char byte = 0;
    int status = read(pipe_fds[0], &byte, 1);
    if (status < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
    {
        throw_errno();
    }
    return status;
}

void semaphore_fd::stop()
{
    if (pipe_fds[1] != -1)
    {
        int status = close(pipe_fds[1]);
        pipe_fds[1] = -1;
        if (status == -1)
            throw_errno();
    }
}

int semaphore_fd::get_fd() const
{
    return pipe_fds[0];
}

} // namespace spead2
