/**
 * @file
 */

#include <cmath>
#include <stdexcept>
#include "send_stream.h"

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

} // namespace send
} // namespace spead2
