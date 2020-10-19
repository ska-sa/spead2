/* Copyright 2015, 2019-2020 National Research Foundation (SARAO)
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

#ifndef SPEAD2_SEND_STREAMBUF_H
#define SPEAD2_SEND_STREAMBUF_H

#include <streambuf>
#include <functional>
#include <boost/asio.hpp>
#include <spead2/send_stream.h>
#include <spead2/send_writer.h>

namespace spead2
{
namespace send
{

/**
 * Puts packets into a streambuf (which could come from an @c ostream). This
 * should not be used for a blocking stream such as a wrapper around TCP,
 * because doing so will block the asio handler thread.
 */
class streambuf_stream : public stream
{
public:
    /// Constructor
    streambuf_stream(
        io_service_ref io_service,
        std::streambuf &streambuf,
        const stream_config &config = stream_config());
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_STREAMBUF_H
