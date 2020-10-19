/* Copyright 2016, 2019-2020 National Research Foundation (SARAO)
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

#ifndef SPEAD2_SEND_UDP_IBV_H
#define SPEAD2_SEND_UDP_IBV_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#if SPEAD2_USE_IBV

#include <boost/asio.hpp>
#include <vector>
#include <utility>
#include <initializer_list>
#include <spead2/send_stream.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_ibv.h>

namespace spead2
{

// Prevent the compiler instantiating the template in all translation units
// (we'll explicitly instantiate it in send_udp_ibv.cpp).
namespace send { class udp_ibv_config; }
extern template class detail::udp_ibv_config_base<send::udp_ibv_config>;

namespace send
{

/**
 * Configuration for @ref udp_ibv_stream.
 */
class udp_ibv_config : public spead2::detail::udp_ibv_config_base<udp_ibv_config>
{
public:
    typedef std::pair<const void *, std::size_t> memory_region;

    /// Default send buffer size
    static constexpr std::size_t default_buffer_size = 512 * 1024;
    /// Default number of times to poll in a row
    static constexpr int default_max_poll = 10;

private:
    friend class spead2::detail::udp_ibv_config_base<udp_ibv_config>;
    static void validate_endpoint(const boost::asio::ip::udp::endpoint &endpoint);
    static void validate_memory_region(const udp_ibv_config::memory_region &region);

    std::uint8_t ttl = 1;
    std::vector<memory_region> memory_regions;

public:
    /// Get the IP TTL
    std::uint8_t get_ttl() const { return ttl; }
    /// Set the IP TTL
    udp_ibv_config &set_ttl(std::uint8_t ttl);

    /// Get currently registered memory regions
    const std::vector<memory_region> &get_memory_regions() const { return memory_regions; }
    /**
     * Register a set of memory regions (replacing any previous). Items stored
     * inside such pre-registered memory regions can (in most cases) be
     * transmitted without making a copy. A memory region is defined by a
     * start pointer and a size in bytes.
     *
     * Memory regions must not overlap; this is only validating when constructing
     * the stream.
     */
    udp_ibv_config &set_memory_regions(const std::vector<memory_region> &memory_regions);
    /// Append a memory region (see @ref set_memory_regions)
    udp_ibv_config &add_memory_region(const void *ptr, std::size_t size);
};

class udp_ibv_stream : public stream
{
public:
    SPEAD2_DEPRECATED("use spead2::send::udp_ibv_config::default_buffer_size")
    static constexpr std::size_t default_buffer_size = udp_ibv_config::default_buffer_size;
    SPEAD2_DEPRECATED("use spead2::send::udp_ibv_config::default_max_poll")
    static constexpr int default_max_poll = udp_ibv_config::default_max_poll;

    /**
     * Backwards-compatibility constructor (taking only a single endpoint).
     *
     * Refer to @ref udp_ibv_config for an explanation of the arguments.
     *
     * @throws std::invalid_argument if @a endpoint is not an IPv4 multicast address
     * @throws std::invalid_argument if @a interface_address is not an IPv4 address
     */
    SPEAD2_DEPRECATED("use udp_ibv_config")
    udp_ibv_stream(
        io_service_ref io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        const boost::asio::ip::address &interface_address,
        std::size_t buffer_size = udp_ibv_config::default_buffer_size,
        int ttl = 1,
        int comp_vector = 0,
        int max_poll = udp_ibv_config::default_max_poll);

    /**
     * Constructor.
     *
     * @param io_service   I/O service for sending data
     * @param config       Common stream configuration
     * @param ibv_config   Class-specific stream configuration
     *
     * @throws std::invalid_argument if @a ibv_config does not have an interface address set.
     * @throws std::invalid_argument if @a ibv_config does not have any endpoints set.
     * @throws std::invalid_argument if memory regions overlap.
     */
    udp_ibv_stream(
        io_service_ref io_service,
        const stream_config &config,
        const udp_ibv_config &ibv_config);
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_USE_IBV
#endif // SPEAD2_SEND_UDP_IBV_H
