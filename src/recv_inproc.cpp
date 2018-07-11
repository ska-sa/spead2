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

#include <cstddef>
#include <memory>
#include <functional>
#include <spead2/common_semaphore.h>
#include <spead2/common_logging.h>
#include <spead2/send_packet.h>
#include <spead2/recv_inproc.h>
#include <spead2/recv_reader.h>

namespace spead2
{
namespace recv
{

inproc_reader::inproc_reader(
    stream &owner,
    std::shared_ptr<ringbuffer<spead2::send::packet, semaphore_fd, semaphore_fd>> queue)
    : reader(owner),
    queue(std::move(queue)),
    data_sem_wrapper(wrap_fd(get_io_service(), this->queue->get_data_sem().get_fd()))
{
    enqueue();
}

void inproc_reader::process_one_packet(const spead2::send::packet &packet)
{
    // The sender always sends the packet as a single piece
    assert(packet.buffers.size() == 1);
    packet_header header;
    auto data = boost::asio::buffer_cast<const std::uint8_t *>(packet.buffers[0]);
    std::size_t length = boost::asio::buffer_size(packet.buffers[0]);
    std::size_t size = decode_packet(header, data, length);
    if (size == length)
    {
        get_stream_base().add_packet(header);
    }
    else if (size != 0)
    {
        log_info("discarding packet due to size mismatch (%1% != %2%)", size, length);
    }
}

void inproc_reader::packet_handler(
    const boost::system::error_code &error,
    std::size_t bytes_transferred)
{
    if (get_stream_base().is_stopped())
    {
        log_info("inproc reader: discarding packet received after stream stopped");
    }
    else
    {
        try
        {
            spead2::send::packet packet = queue->try_pop();
            process_one_packet(packet);
        }
        catch (ringbuffer_stopped)
        {
            get_stream_base().stop_received();
        }
        catch (ringbuffer_empty)
        {
            // spurious wakeup - no action needed
        }
    }
    if (!get_stream_base().is_stopped())
        enqueue();
    else
    {
        data_sem_wrapper.close();
        stopped();
    }
}

void inproc_reader::enqueue()
{
    using namespace std::placeholders;
    data_sem_wrapper.async_read_some(
        boost::asio::null_buffers(),
        get_stream().get_strand().wrap(std::bind(&inproc_reader::packet_handler, this, _1, _2)));
}

void inproc_reader::stop()
{
    data_sem_wrapper.close();
}

bool inproc_reader::lossy() const
{
    return false;
}

} // namespace recv
} // namespace spead2
