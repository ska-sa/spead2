/* Copyright 2020 National Research Foundation (SARAO)
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

#include <stdexcept>
#include <cmath>
#include <spead2/send_stream_config.h>

namespace spead2
{
namespace send
{

constexpr std::size_t stream_config::default_max_packet_size;
constexpr std::size_t stream_config::default_max_heaps;
constexpr std::size_t stream_config::default_burst_size;
constexpr double stream_config::default_burst_rate_ratio;
constexpr rate_method stream_config::default_rate_method;

stream_config &stream_config::set_max_packet_size(std::size_t max_packet_size)
{
    // TODO: validate here instead rather than waiting until packet_generator
    this->max_packet_size = max_packet_size;
    return *this;
}

stream_config &stream_config::set_rate(double rate)
{
    if (rate < 0.0 || !std::isfinite(rate))
        throw std::invalid_argument("rate must be non-negative");
    this->rate = rate;
    return *this;
}

stream_config &stream_config::set_max_heaps(std::size_t max_heaps)
{
    if (max_heaps == 0)
        throw std::invalid_argument("max_heaps must be positive");
    this->max_heaps = max_heaps;
    return *this;
}

stream_config &stream_config::set_burst_size(std::size_t burst_size)
{
    this->burst_size = burst_size;
    return *this;
}

stream_config &stream_config::set_burst_rate_ratio(double burst_rate_ratio)
{
    if (burst_rate_ratio < 1.0 || !std::isfinite(burst_rate_ratio))
        throw std::invalid_argument("burst rate ratio must be at least 1.0 and finite");
    this->burst_rate_ratio = burst_rate_ratio;
    return *this;
}

stream_config &stream_config::set_rate_method(rate_method method)
{
    this->method = method;
    return *this;
}

double stream_config::get_burst_rate() const
{
    return rate * burst_rate_ratio;
}

stream_config::stream_config()
{
}

} // namespace send
} // namespace spead2
