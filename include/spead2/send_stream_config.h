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

#ifndef SPEAD2_SEND_STREAM_CONFIG_H
#define SPEAD2_SEND_STREAM_CONFIG_H

#include <cstddef>

namespace spead2
{

namespace send
{

enum class rate_method
{
    SW,        ///< Software rate limiter
    HW,        ///< Hardware rate limiter, if available
    AUTO       ///< Implementation decides on rate-limit method
};

/**
 * Configuration for send streams.
 */
class stream_config
{
public:
    static constexpr std::size_t default_max_packet_size = 1472;
    static constexpr std::size_t default_max_heaps = 4;
    static constexpr std::size_t default_burst_size = 65536;
    static constexpr double default_burst_rate_ratio = 1.05;
    static constexpr rate_method default_rate_method = rate_method::AUTO;

    /// Set maximum packet size to use (only counts the UDP payload, not L1-4 headers).
    stream_config &set_max_packet_size(std::size_t max_packet_size);
    /// Get maximum packet size to use.
    std::size_t get_max_packet_size() const { return max_packet_size; }
    /// Set maximum transmit rate to use, in bytes per second.
    stream_config &set_rate(double rate);
    /// Get maximum transmit rate to use, in bytes per second.
    double get_rate() const { return rate; }
    /// Set maximum size of a burst, in bytes.
    stream_config &set_burst_size(std::size_t burst_size);
    /// Get maximum size of a burst, in bytes.
    std::size_t get_burst_size() const { return burst_size; }
    /// Set maximum number of in-flight heaps.
    stream_config &set_max_heaps(std::size_t max_heaps);
    /// Get maximum number of in-flight heaps.
    std::size_t get_max_heaps() const { return max_heaps; }
    /// Set maximum increase in transmit rate for catching up.
    stream_config &set_burst_rate_ratio(double burst_rate_ratio);
    /// Get maximum increase in transmit rate for catching up.
    double get_burst_rate_ratio() const { return burst_rate_ratio; }
    /// Set rate-limiting method
    stream_config &set_rate_method(rate_method method);
    /// Get rate-limiting method
    rate_method get_rate_method() const { return method; }

    /// Get product of rate and burst_rate_ratio
    double get_burst_rate() const;

    stream_config();

private:
    std::size_t max_packet_size = default_max_packet_size;
    double rate = 0.0;
    std::size_t burst_size = default_burst_size;
    std::size_t max_heaps = default_max_heaps;
    double burst_rate_ratio = default_burst_rate_ratio;
    rate_method method = default_rate_method;
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_STREAM_CONFIG_H
