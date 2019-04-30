/* Copyright 2018-2019 SKA South Africa
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
#include <spead2/common_inproc.h>
#include <spead2/common_logging.h>
#include <spead2/recv_inproc.h>
#include <spead2/recv_reader.h>

namespace spead2
{
namespace recv
{

inproc_reader::inproc_reader(
    stream &owner,
    std::shared_ptr<inproc_queue> queue)
    : reader(owner),
    queue(std::move(queue)),
    data_sem_wrapper(wrap_fd(owner.get_io_service(),
                             this->queue->buffer.get_data_sem().get_fd()))
{
    enqueue();
}

void inproc_reader::process_one_packet(stream_base::add_packet_state &state,
                                       const inproc_queue::packet &packet)
{
    packet_header header;
    std::size_t size = decode_packet(header, packet.data.get(), packet.size);
    if (size == packet.size)
    {
        state.add_packet(header);
    }
    else if (size != 0)
    {
        log_info("discarding packet due to size mismatch (%1% != %2%)", size, packet.size);
    }
}

void inproc_reader::packet_handler(
    const boost::system::error_code &error,
    std::size_t bytes_transferred)
{
    stream_base::add_packet_state state(get_stream_base());
    if (!error)
    {
        if (state.is_stopped())
        {
            log_info("inproc reader: discarding packet received after stream stopped");
        }
        else
        {
            try
            {
                inproc_queue::packet packet = queue->buffer.try_pop();
                process_one_packet(state, packet);
                /* TODO: could grab a batch of packets to amortise costs */
            }
            catch (ringbuffer_stopped &)
            {
                state.stop();
            }
            catch (ringbuffer_empty &)
            {
                // spurious wakeup - no action needed
            }
        }
    }
    else if (error != boost::asio::error::operation_aborted)
        log_warning("Error in inproc receiver: %1%", error.message());

    if (!state.is_stopped())
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
        std::bind(&inproc_reader::packet_handler, this, _1, _2));
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
