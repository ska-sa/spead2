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

/**
 * @file
 */

#ifndef SPEAD2_SEND_INPROC_H
#define SPEAD2_SEND_INPROC_H

#include <cstddef>
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

    template<typename Handler>
    void async_send_packet(const packet &pkt, Handler &&handler)
    {
        auto callback = [this, &pkt, handler](const boost::system::error_code &ec,
                                              std::size_t bytes_transferred)
        {
            if (!ec)
            {
                inproc_queue::packet dup = detail::copy_packet(pkt);
                try
                {
                    queue->buffer.try_push(std::move(dup));
                }
                catch (ringbuffer_full)
                {
                    // Another thread in the thundering herd beat us
                    // Schedule to try again.
                    async_send_packet(pkt, std::move(handler));
                    return;
                }
                std::size_t size = boost::asio::buffer_size(pkt.buffers);
                handler(boost::system::error_code(), size);
            }
            else
            {
                handler(ec, 0);
            }
        };

        space_sem_wrapper.async_read_some(boost::asio::null_buffers(), callback);
    }

    std::shared_ptr<inproc_queue> queue;
    boost::asio::posix::stream_descriptor space_sem_wrapper;

public:
    /// Constructor
    inproc_stream(
        io_service_ref io_service,
        std::shared_ptr<inproc_queue> queue,
        const stream_config &config = stream_config());
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_INPROC_H
