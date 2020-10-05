/* Copyright 2016, 2019-2020 SKA South Africa
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

namespace spead2
{
namespace send
{

/**
 * Configuration for @ref udp_ibv_stream.
 */
class udp_ibv_stream_config
{
public:
    typedef std::pair<const void *, std::size_t> memory_region;

private:
    std::vector<boost::asio::ip::udp::endpoint> endpoints;
    boost::asio::ip::address interface_address;
    std::size_t buffer_size;
    std::uint8_t ttl;
    int comp_vector;
    int max_poll;
    std::vector<memory_region> memory_regions;

public:
    /**
     * Construct with the defaults.
     *
     * It cannot be used as is, because the interface address and at least one
     * endpoint must be supplied.
     */
    udp_ibv_stream_config();

    /// Get the configured endpoints
    const std::vector<boost::asio::ip::udp::endpoint> &get_endpoints() const { return endpoints; }
    /**
     * Set all endpoints at once. Each endpoint corresponds to a substream.
     *
     * @throws std::invalid_argument if any element of @a endpoints is not an IPv4 multicast address.
     */
    udp_ibv_stream_config &set_endpoints(const std::vector<boost::asio::ip::udp::endpoint> &endpoints);
    /**
     * Append a single endpoint.
     *
     * @throws std::invalid_argument if @a endpoint is not an IPv4 multicast address.
     */
    udp_ibv_stream_config &add_endpoint(const boost::asio::ip::udp::endpoint &endpoint);

    /// Get the currently set interface address
    const boost::asio::ip::address get_interface_address() const { return interface_address; }
    /**
     * Set the interface address.
     *
     * @throws std::invalid_argument if @a interface_address is not an IPv4 address.
     */
    udp_ibv_stream_config &set_interface_address(const boost::asio::ip::address &interface_address);

    /// Get the currently configured buffer size.
    std::size_t get_buffer_size() const { return buffer_size; }
    /**
     * Set the buffer size.
     *
     * The value 0 is special and resets it to the default. The actual buffer size
     * used may be slightly different to round it to a whole number of
     * packet-sized slots.
     */
    udp_ibv_stream_config &set_buffer_size(std::size_t buffer_size);

    /// Get the IP TTL
    std::uint8_t get_ttl() const { return ttl; }
    /// Set the IP TTL
    udp_ibv_stream_config &set_ttl(std::uint8_t ttl);

    /// Get the completion channel vector (see @ref set_comp_vector)
    int get_comp_vector() const { return comp_vector; }
    /**
     * Set the completion channel vector (interrupt) for asynchronous operation.
     * Use a negative value to poll continuously. Polling should not be used if
     * there are other users of the thread pool. If a non-negative value is
     * provided, it is taken modulo the number of available completion vectors.
     * This allows a number of readers to be assigned sequential completion
     * vectors and have them load-balanced, without concern for the number
     * available.
     */
    udp_ibv_stream_config &set_comp_vector(int comp_vector);

    /// Get maximum number of times to poll in a row (see @ref set_max_poll)
    int get_max_poll() const { return max_poll; }
    /**
     * Set maximum number of times to poll in a row.
     *
     * If interrupts are enabled (default), it is the maximum number of times
     * to poll before waiting for an interrupt; if they are disabled by @ref
     * set_comp_vector, it is the number of times to poll before letting other
     * code run on the thread.
     *
     * @throws std::invalid_argument if @a max_poll is zero.
     */
    udp_ibv_stream_config &set_max_poll(int max_poll);

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
    udp_ibv_stream_config &set_memory_regions(const std::vector<memory_region> &memory_regions);
    /// Append a memory region (see @ref set_memory_regions)
    udp_ibv_stream_config &add_memory_region(const void *ptr, std::size_t size);
};

class udp_ibv_stream : public stream
{
public:
    /// Default send buffer size, if none is passed to the constructor
    static constexpr std::size_t default_buffer_size = 512 * 1024;
    /// Number of times to poll in a row, if none is explicitly passed to the constructor
    static constexpr int default_max_poll = 10;

    /**
     * Backwards-compatibility constructor (taking only a single endpoint).
     *
     * Refer to @ref udp_ibv_stream_config for an explanation of the arguments.
     *
     * @throws std::invalid_argument if @a endpoint is not an IPv4 multicast address
     * @throws std::invalid_argument if @a interface_address is not an IPv4 address
     */
    SPEAD2_DEPRECATED("use udp_ibv_stream_config")
    udp_ibv_stream(
        io_service_ref io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        const boost::asio::ip::address &interface_address,
        std::size_t buffer_size = default_buffer_size,
        int ttl = 1,
        int comp_vector = 0,
        int max_poll = default_max_poll);

    /**
     * Constructor.
     *
     * @param io_service   I/O service for sending data
     * @param config       Common stream configuration
     * @param udp_ibv_config  Class-specific stream configuration
     *
     * @throws std::invalid_argument if @a udp_ibv_config does not an interface address set.
     * @throws std::invalid_argument if @a udp_ibv_config does not have any endpoints set.
     * @throws std::invalid_argument if memory regions overlap.
     */
    udp_ibv_stream(
        io_service_ref io_service,
        const stream_config &config,
        const udp_ibv_stream_config &udp_ibv_config);
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_USE_IBV
#endif // SPEAD2_SEND_UDP_IBV_H
