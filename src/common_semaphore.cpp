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
#include "common_semaphore.h"
#include "common_logging.h"

namespace spead2
{

semaphore::semaphore()
{
    if (pipe(pipe_fds) == -1)
        throw std::system_error(errno, std::system_category());
    for (int i = 0; i < 2; i++)
    {
        int flags = fcntl(pipe_fds[i], F_GETFD);
        if (flags == -1)
            throw std::system_error(errno, std::system_category());
        flags |= FD_CLOEXEC;
        if (fcntl(pipe_fds[i], F_SETFD, flags) == -1)
            throw std::system_error(errno, std::system_category());
    }
}

semaphore::~semaphore()
{
    for (int i = 0; i < 2; i++)
        if (pipe_fds[i] != -1)
        {
            if (close(pipe_fds[i]) == -1)
            {
                // Can't throw, because this is a destructor
                std::error_code code(errno, std::system_category());
                log_warning("failed to close pipe: %1% (%2%)", code.value(), code.message());
            }
        }
}

void semaphore::put()
{
    char byte = 0;
    int status;
    do
    {
        status = write(pipe_fds[1], &byte, 1);
        if (status < 0 && errno != EINTR)
        {
            throw std::system_error(errno, std::system_category());
        }
    } while (status < 0);
}

int semaphore::get()
{
    char byte = 0;
    int status = read(pipe_fds[0], &byte, 1);
    if (status < 0 && errno != EINTR)
    {
        throw std::system_error(errno, std::system_category());
    }
    return status;
}

void semaphore::stop()
{
    if (pipe_fds[1] != -1)
    {
        int status = close(pipe_fds[1]);
        pipe_fds[1] = -1;
        if (status == -1)
            throw std::system_error(errno, std::system_category());
    }
}

int semaphore::get_fd() const
{
    return pipe_fds[0];
}

} // namespace spead2
