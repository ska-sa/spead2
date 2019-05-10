/* Copyright 2015, 2019 SKA South Africa
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

#include <streambuf>
#include <spead2/send_streambuf.h>

namespace spead2
{
namespace send
{

void streambuf_stream::async_send_packets()
{
    for (std::size_t i = 0; i < n_current_packets; i++)
    {
        for (const auto &buffer : current_packets[i].pkt.buffers)
        {
            std::size_t buffer_size = boost::asio::buffer_size(buffer);
            // TODO: handle errors
            streambuf.sputn(boost::asio::buffer_cast<const char *>(buffer), buffer_size);
        }
        current_packets[i].result = boost::system::error_code();
    }
    get_io_service().post([this] { packets_handler(); });
}

streambuf_stream::streambuf_stream(
    io_service_ref io_service,
    std::streambuf &streambuf,
    const stream_config &config)
    : stream_impl<streambuf_stream>(std::move(io_service), config, 64), streambuf(streambuf)
{
}

streambuf_stream::~streambuf_stream()
{
    flush();
}

} // namespace send
} // namespace spead2
