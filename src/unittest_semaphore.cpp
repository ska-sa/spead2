/* Copyright 2019 SKA South Africa
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
 * Unit tests for common_semaphore.
 */

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <spead2/common_semaphore.h>
#include <future>
#include <utility>
#include <cstring>
#include <cerrno>
#include <poll.h>

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(common)
BOOST_AUTO_TEST_SUITE(semaphore)

typedef boost::mpl::list<
    spead2::semaphore_spin,
    spead2::semaphore_pipe,
#if SPEAD2_USE_EVENTFD
    spead2::semaphore_eventfd,
#endif
#if SPEAD2_USE_POSIX_SEMAPHORES
    spead2::semaphore_posix,
#endif
    spead2::semaphore_fd,
    spead2::semaphore> semaphore_types;

typedef boost::mpl::list<
#if SPEAD2_USE_EVENTFD
    spead2::semaphore_eventfd,
#endif
    spead2::semaphore_pipe> semaphore_fd_types;

/* Try to get a semaphore, but return only if it's zero, not on an interrupted
 * system call.
 */
template<typename T>
static int semaphore_try_get(T &sem)
{
    while (true)
    {
        errno = 0;
        int result = sem.try_get();
        if (result != -1 || errno != EINTR)
            return result;
    }
}

/* Poll that restarts after interrupted system calls (but does not try to
 * adjust the timeout to compensate).
 */
static int poll_restart(struct pollfd *fds, nfds_t nfds, int timeout)
{
    while (true)
    {
        int result = poll(fds, nfds, timeout);
        if (result >= 0 || errno != EINTR)
            return result;
    }
}

/* Gets a semaphore until it would block, to determine what value it had.
 * It does not restore the previous value.
 */
template<typename T>
static int semaphore_get_value(T &sem)
{
    int value = 0;
    int result;
    while ((result = semaphore_try_get(sem)) == 0)
        value++;
    BOOST_CHECK_EQUAL(result, -1);
    return value;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(single_thread, T, semaphore_types)
{
    T sem(2);
    BOOST_CHECK_EQUAL(semaphore_get_value(sem), 2);
    sem.put();
    BOOST_CHECK_EQUAL(semaphore_get_value(sem), 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(multi_thread, T, semaphore_types)
{
    const std::int64_t N = 100000;
    std::vector<int> x(N);
    T sem(0);
    /* Have a separate thread write to x and put a semaphore each time it
     * does, and this thread get the semaphore before reading a value. This
     * doesn't necessarily prove that anything works (that's basically
     * impossible for multi-threading primitives), but it can show up
     * failures.
     */
    auto worker = [&x, &sem, N] {
        for (int i = 0; i < N; i++)
        {
            x[i] = i;
            sem.put();
        }
    };
    auto future = std::async(std::launch::async, worker);
    std::int64_t sum = 0;
    for (int i = 0; i < N; i++)
    {
        semaphore_get(sem);
        sum += x[i];
    }
    future.get();
    BOOST_CHECK_EQUAL(sum, N * (N - 1) / 2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(move_assign, T, semaphore_fd_types)
{
    T sem1(2);
    int orig_fd = sem1.get_fd();
    T sem2;
    sem2 = std::move(sem1);
    BOOST_CHECK_EQUAL(sem2.get_fd(), orig_fd);
    BOOST_CHECK_EQUAL(semaphore_get_value(sem2), 2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(move_construct, T, semaphore_fd_types)
{
    T sem1(2);
    int orig_fd = sem1.get_fd();
    T sem2(std::move(sem1));
    BOOST_CHECK_EQUAL(sem2.get_fd(), orig_fd);
    BOOST_CHECK_EQUAL(semaphore_get_value(sem2), 2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(poll_fd, T, semaphore_fd_types)
{
    T sem(1);
    pollfd fds[1];
    std::memset(&fds, 0, sizeof(fds));
    fds[0].fd = sem.get_fd();
    fds[0].events = POLLIN;
    int result = poll_restart(fds, 1, 0);
    BOOST_CHECK_EQUAL(result, 1);
    semaphore_get(sem);
    result = poll_restart(fds, 1, 1);
    BOOST_CHECK_EQUAL(result, 0);
}

BOOST_AUTO_TEST_SUITE_END()  // semaphore
BOOST_AUTO_TEST_SUITE_END()  // common

}} // namespace spead2::unittest
