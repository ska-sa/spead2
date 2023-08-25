/* Copyright 2018-2020, 2023 National Research Foundation (SARAO)
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

#include <vector>
#include <memory>
#include <boost/asio.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/common_inproc.h>
#include <spead2/send_stream.h>

namespace spead2::send
{

class inproc_stream : public stream
{
public:
    /// Constructor
    inproc_stream(
        io_service_ref io_service,
        const std::vector<std::shared_ptr<inproc_queue>> &queues,
        const stream_config &config = stream_config());

    /// Get the underlying storage queues
    const std::vector<std::shared_ptr<inproc_queue>> &get_queues() const;
};

} // namespace spead2::send

#endif // SPEAD2_SEND_INPROC_H
