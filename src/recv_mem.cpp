/* Copyright 2015, 2019, 2023, 2025 National Research Foundation (SARAO)
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

#include <cstdint>
#include <cassert>
#include <spead2/recv_mem.h>
#include <spead2/recv_stream.h>

namespace spead2::recv
{

mem_reader::mem_reader(
    stream &owner,
    const std::uint8_t *ptr, std::size_t length)
    : reader(owner), ptr(ptr), length(length)
{
    assert(ptr != nullptr);
}

void mem_reader::start()
{
    boost::asio::post(
        get_io_context(),
        bind_handler([this] (handler_context, stream_base::add_packet_state &state) {
            mem_to_stream(state, this->ptr, this->length);
            // There will be no more data, so we can stop the stream immediately.
            state.stop();
        })
    );
}

bool mem_reader::lossy() const
{
    return false;
}

} // namespace spead2::recv
