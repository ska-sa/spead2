/* Copyright 2015, 2020 National Research Foundation (SARAO)
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
#include <utility>
#include <spead2/recv_ring_stream.h>

namespace spead2
{
namespace recv
{

constexpr std::size_t ring_stream_config::default_heaps;

ring_stream_config &ring_stream_config::set_heaps(std::size_t heaps)
{
    if (heaps == 0)
        throw std::invalid_argument("heaps must be at least 1");
    this->heaps = heaps;
    return *this;
}

ring_stream_config &ring_stream_config::set_contiguous_only(bool contiguous_only)
{
    this->contiguous_only = contiguous_only;
    return *this;
}

ring_stream_base::ring_stream_base(
    io_service_ref io_service,
    const stream_config &config,
    const ring_stream_config &ring_config)
    : stream(std::move(io_service), config),
    ring_config(ring_config)
{
}

} // namespace recv
} // namespace spead2
