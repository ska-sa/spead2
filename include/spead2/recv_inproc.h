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

#ifndef SPEAD2_RECV_INPROC_H
#define SPEAD2_RECV_INPROC_H

#include <memory>
#include <boost/asio.hpp>
#include <spead2/common_inproc.h>
#include <spead2/recv_reader.h>
#include <spead2/recv_stream.h>

namespace spead2
{
namespace recv
{

/**
 * Stream reader that receives packets from an @ref inproc_queue.
 */
class inproc_reader : public reader
{
private:
    std::shared_ptr<inproc_queue> queue;
    boost::asio::posix::stream_descriptor data_sem_wrapper;

    void process_one_packet(stream_base::add_packet_state &state,
                            const inproc_queue::packet &packet);
    void packet_handler(const boost::system::error_code &error, std::size_t bytes_received);
    void enqueue();

public:
    /// Constructor.
    inproc_reader(
        stream &owner,
        std::shared_ptr<inproc_queue> queue);

    virtual void stop() override;
    virtual bool lossy() const override;
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_INPROC_H
