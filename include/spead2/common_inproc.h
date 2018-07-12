/* Copyright 2018 SKA South Africa
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

#ifndef SPEAD2_COMMON_INPROC_H
#define SPEAD2_COMMON_INPROC_H

#include <memory>
#include <cstdint>
#include <cstddef>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_semaphore.h>

namespace spead2
{

/**
 * Queue for packets being passed within the process.
 *
 * While the members are public, this is only to allow the send and receive
 * code (and unit tests) to access the data. Users are advised to treat this
 * class as opaque.
 */
class inproc_queue
{
public:
    struct packet
    {
        std::unique_ptr<std::uint8_t[]> data;
        std::size_t size;
    };

    ringbuffer<packet, semaphore_fd, semaphore_fd> buffer;

    explicit inproc_queue(std::size_t capacity);
};

extern template class ringbuffer<inproc_queue::packet, semaphore_fd, semaphore_fd>;

} // namespace spead2

#endif // SPEAD2_COMMON_INPROC_H
