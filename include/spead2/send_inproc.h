/* Copyright 2018-2020 SKA South Africa
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
 */

#ifndef SPEAD2_SEND_INPROC_H
#define SPEAD2_SEND_INPROC_H

#include <cstddef>
#include <vector>
#include <utility>
#include <memory>
#include <boost/asio.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_inproc.h>
#include <spead2/send_packet.h>
#include <spead2/send_stream.h>

namespace spead2
{
namespace send
{

namespace detail
{

/// Create a copy of a packet that owns all its own data
inproc_queue::packet copy_packet(const packet &in);

} // namespace detail

class inproc_stream : public stream_impl<inproc_stream>
{
private:
    friend class stream_impl<inproc_stream>;
    std::vector<std::shared_ptr<inproc_queue>> queues;

    void async_send_packets();

public:
    /// Backwards-compatibility constructor (taking only a single queue)
    SPEAD2_DEPRECATED("use a vector of queues")
    inproc_stream(
        io_service_ref io_service,
        std::shared_ptr<inproc_queue> queue,
        const stream_config &config = stream_config());

    /// Constructor
    inproc_stream(
        io_service_ref io_service,
        const std::vector<std::shared_ptr<inproc_queue>> &queues,
        const stream_config &config = stream_config());

    /// Get the underlying storage queue
    const std::vector<std::shared_ptr<inproc_queue>> &get_queues() const;

    virtual std::size_t get_num_substreams() const override final { return queues.size(); }

    virtual ~inproc_stream();
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_INPROC_H
