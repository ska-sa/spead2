/* Copyright 2018-2020 National Research Foundation (SARAO)
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
#include <spead2/send_writer.h>

namespace spead2
{
namespace send
{

namespace
{

static inproc_queue::packet copy_packet(const packet &in)
{
    std::size_t size = boost::asio::buffer_size(in.buffers);
    inproc_queue::packet out;
    out.data.reset(new std::uint8_t[size]);
    out.size = size;
    boost::asio::mutable_buffer buffer(out.data.get(), size);
    boost::asio::buffer_copy(buffer, in.buffers);
    return out;
}

class inproc_writer : public writer
{
private:
    std::vector<std::shared_ptr<inproc_queue>> queues;

    virtual void wakeup() override;

public:
    /// Constructor
    inproc_writer(
        io_service_ref io_service,
        const std::vector<std::shared_ptr<inproc_queue>> &queues,
        const stream_config &config);

    /// Get the underlying storage queue
    const std::vector<std::shared_ptr<inproc_queue>> &get_queues() const;

    virtual std::size_t get_num_substreams() const override final { return queues.size(); }
};

void inproc_writer::wakeup()
{
    transmit_packet data;
    switch (get_packet(data))
    {
    case packet_result::SLEEP:
        sleep();
        return;
    case packet_result::EMPTY:
        request_wakeup();
        return;
    case packet_result::SUCCESS:
        break;
    }

    inproc_queue::packet dup = copy_packet(data.pkt);
    std::size_t size = dup.size;
    stream::queue_item *item = data.item;
    try
    {
        queues[data.item->substream_index]->buffer.push(std::move(dup));
        item->bytes_sent += size;
    }
    catch (ringbuffer_stopped &)
    {
        item->result = boost::asio::error::operation_aborted;
    }
    if (data.last)
        heaps_completed(1);
    post_wakeup();
}

const std::vector<std::shared_ptr<inproc_queue>> &inproc_writer::get_queues() const
{
    return queues;
}

inproc_writer::inproc_writer(
    io_service_ref io_service,
    const std::vector<std::shared_ptr<inproc_queue>> &queues,
    const stream_config &config)
    : writer(std::move(io_service), config),
    queues(queues)
{
    if (queues.empty())
        throw std::invalid_argument("queues is empty");
}

} // anonymous namespace

inproc_stream::inproc_stream(
    io_service_ref io_service,
    std::shared_ptr<inproc_queue> queue,
    const stream_config &config)
    : inproc_stream(
        std::move(io_service),
        std::vector<std::shared_ptr<inproc_queue>>{std::move(queue)},
        config)
{
}

inproc_stream::inproc_stream(
    io_service_ref io_service,
    std::initializer_list<std::shared_ptr<inproc_queue>> queues,
    const stream_config &config)
    : inproc_stream(
        std::move(io_service),
        std::vector<std::shared_ptr<inproc_queue>>(queues),
        config)
{
}

inproc_stream::inproc_stream(
    io_service_ref io_service,
    const std::vector<std::shared_ptr<inproc_queue>> &queues,
    const stream_config &config)
    : stream(std::unique_ptr<writer>(new inproc_writer(std::move(io_service), queues, config)))
{
}

const std::vector<std::shared_ptr<inproc_queue>> &inproc_stream::get_queues() const
{
    return static_cast<const inproc_writer &>(get_writer()).get_queues();
}

const std::shared_ptr<inproc_queue> &inproc_stream::get_queue() const
{
    const auto &queues = get_queues();
    if (queues.size() != 1)
        throw std::runtime_error("get_queue only works when there is a single queue. Use get_queues instead");
    return queues[0];
}

} // namespace send
} // namespace spead2
