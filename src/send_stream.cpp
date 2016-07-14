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

#include <cmath>
#include <stdexcept>
#include <spead2/send_stream.h>

namespace spead2
{
namespace send
{

constexpr std::size_t stream_config::default_max_packet_size;
constexpr std::size_t stream_config::default_max_heaps;
constexpr std::size_t stream_config::default_burst_size;

void stream_config::set_max_packet_size(std::size_t max_packet_size)
{
    // TODO: validate here instead rather than waiting until packet_generator
    this->max_packet_size = max_packet_size;
}

std::size_t stream_config::get_max_packet_size() const
{
    return max_packet_size;
}

void stream_config::set_rate(double rate)
{
    if (rate < 0.0 || !std::isfinite(rate))
        throw std::invalid_argument("rate must be non-negative");
    this->rate = rate;
}

double stream_config::get_rate() const
{
    return rate;
}

void stream_config::set_max_heaps(std::size_t max_heaps)
{
    if (max_heaps == 0)
        throw std::invalid_argument("max_heaps must be positive");
    this->max_heaps = max_heaps;
}

std::size_t stream_config::get_max_heaps() const
{
    return max_heaps;
}

void stream_config::set_burst_size(std::size_t burst_size)
{
    this->burst_size = burst_size;
}

std::size_t stream_config::get_burst_size() const
{
    return burst_size;
}

stream_config::stream_config(
    std::size_t max_packet_size,
    double rate,
    std::size_t burst_size,
    std::size_t max_heaps)
{
    set_max_packet_size(max_packet_size);
    set_rate(rate);
    set_burst_size(burst_size);
    set_max_heaps(max_heaps);
}


stream::stream(boost::asio::io_service &io_service)
    : io_service(io_service)
{
}

stream::~stream()
{
}

} // namespace send
} // namespace spead2
