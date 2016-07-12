/* Copyright 2015 SKA South Africa
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
#include <spead2/recv_reader.h>
#include <spead2/recv_mem.h>
#include <spead2/recv_stream.h>

namespace spead2
{
namespace recv
{

mem_reader::mem_reader(
    stream &owner,
    const std::uint8_t *ptr, std::size_t length)
    : reader(owner), ptr(ptr), length(length)
{
    assert(ptr != nullptr);
    get_stream().get_strand().post([this] {
        mem_to_stream(get_stream_base(), this->ptr, this->length);
        // There will be no more data, so we can stop the stream immediately.
        get_stream_base().stop_received();
        stopped();
    });
}

} // namespace recv
} // namespace spead2
