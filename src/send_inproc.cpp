/* Copyright 2018, 2019 SKA South Africa
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

#include <cstddef>
#include <utility>
#include <memory>
#include <boost/asio.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/common_inproc.h>
#include <spead2/send_packet.h>
#include <spead2/send_inproc.h>

namespace spead2
{
namespace send
{

namespace detail
{

inproc_queue::packet copy_packet(const packet &in)
{
    std::size_t size = boost::asio::buffer_size(in.buffers);
    inproc_queue::packet out;
    out.data.reset(new std::uint8_t[size]);
    out.size = size;
    boost::asio::mutable_buffer buffer(out.data.get(), size);
    boost::asio::buffer_copy(buffer, in.buffers);
    return out;
}

} // namespace detail

void inproc_stream::async_send_packets()
{
    for (std::size_t i = 0; i < n_current_packets; i++)
    {
        inproc_queue::packet dup = detail::copy_packet(current_packets[i].pkt);
        try
        {
            queue->buffer.push(std::move(dup));
            current_packets[i].result = boost::system::error_code();
        }
        catch (ringbuffer_stopped &)
        {
            current_packets[i].result = boost::asio::error::operation_aborted;
        }
    }
    get_io_service().post([this] { packets_handler(); });
}

inproc_stream::inproc_stream(
    io_service_ref io_service,
    std::shared_ptr<inproc_queue> queue,
    const stream_config &config)
    : stream_impl<inproc_stream>(std::move(io_service), config, 1),
    queue(std::move(queue))
{
}

inproc_stream::~inproc_stream()
{
    flush();
}

std::shared_ptr<inproc_queue> inproc_stream::get_queue() const
{
    return queue;
}

} // namespace send
} // namespace spead2
