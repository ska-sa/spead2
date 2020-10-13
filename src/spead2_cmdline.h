/* Copyright 2020 SKA South Africa
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
 * @file Shared command-line processing for tools.
 */

#ifndef SPEAD2_CMDLINE_H
#define SPEAD2_CMDLINE_H

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <istream>
#include <ostream>
#include <boost/asio.hpp>
#include <boost/program_options.hpp>
#include <boost/optional.hpp>
#include <spead2/common_features.h>
#include <spead2/common_flavour.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_ring_stream.h>
#include <spead2/send_stream.h>
#if SPEAD2_USE_IBV
# include <spead2/recv_udp_ibv.h>
# include <spead2/send_udp_ibv.h>
#endif

namespace spead2
{

template<typename T>
static inline boost::program_options::typed_value<T> *
make_value_semantic(T *out)
{
    return boost::program_options::value<T>(out)->default_value(*out);
}

template<typename T>
static inline boost::program_options::typed_value<boost::optional<T>> *
make_value_semantic(boost::optional<T> *out)
{
    return boost::program_options::value(out);
}

template<typename T>
static inline boost::program_options::typed_value<std::vector<T>> *
make_value_semantic(std::vector<T> *out)
{
    return boost::program_options::value(out)->composing();
}

static inline boost::program_options::typed_value<bool> *
make_value_semantic(bool *out)
{
    return boost::program_options::bool_switch(out);
}

class option_adder
{
private:
    boost::program_options::options_description &desc;
    std::map<std::string, std::string> name_map;

public:
    option_adder(boost::program_options::options_description &desc,
                 const std::map<std::string, std::string> &name_map = {})
        : desc(desc), name_map(name_map)
    {
    }

    template<typename T>
    void operator()(const std::string &name, const std::string &description, T *value) const
    {
        auto it = name_map.find(name);
        std::string new_name = it == name_map.end() ? name : it->second;
        if (!new_name.empty())
        {
            desc.add_options()
                (new_name.c_str(), make_value_semantic(value), description.c_str());
        }
    }
};

struct protocol_options
{
    bool tcp = false;
    bool pyspead = false;

    void notify();

    template<typename T>
    void enumerate(T &&callback)
    {
        callback("tcp", "Use TCP instead of UDP", &tcp);
        callback("pyspead", "Be bug-compatible with PySPEAD", &pyspead);
    }
};

template<typename Proto>
std::vector<boost::asio::ip::basic_endpoint<Proto>> parse_endpoints(
    boost::asio::io_service &io_service, const std::vector<std::string> &endpoints, bool passive);

template<typename Proto>
boost::asio::ip::basic_endpoint<Proto> parse_endpoint(
    boost::asio::io_service &io_service, const std::string &endpoint, bool passive);

template<typename Proto>
std::vector<boost::asio::ip::basic_endpoint<Proto>> parse_endpoints(
    const std::vector<std::string> &endpoints, bool passive);

template<typename Proto>
boost::asio::ip::basic_endpoint<Proto> parse_endpoint(
    const std::string &endpoint, bool passive);


namespace recv
{

struct receiver_options
{
    bool ring = false;
    bool memcpy_nt = false;
    std::size_t max_heaps = stream_config::default_max_heaps;
    std::size_t ring_heaps = ring_stream_config::default_heaps;
    bool mem_pool = false;
    std::size_t mem_lower = 16384;
    std::size_t mem_upper = 32 * 1024 * 1024;
    std::size_t mem_max_free = 12;
    std::size_t mem_initial = 8;
    boost::optional<std::size_t> buffer_size;
    boost::optional<std::size_t> max_packet_size;
    std::string interface_address;
#if SPEAD2_USE_IBV
    bool ibv = false;
    int ibv_comp_vector = 0;
    int ibv_max_poll = spead2::recv::udp_ibv_reader::default_max_poll;
#endif

    void notify(const protocol_options &protocol);

    template<typename T>
    void enumerate(T &&callback)
    {
        callback("bind", "Interface address", &interface_address);
        callback("packet", "Maximum packet size to accept", &max_packet_size);
        callback("buffer", "Socket buffer size", &buffer_size);
        callback("concurrent-heaps", "Maximum number of in-flight heaps", &max_heaps);
        callback("ring-heaps", "Ring buffer capacity in heaps", &ring_heaps);
        callback("mem-pool", "Use a memory pool", &mem_pool);
        callback("mem-lower", "Minimum allocation which will use the memory pool", &mem_lower);
        callback("mem-upper", "Maximum allocation which will use the memory pool", &mem_upper);
        callback("mem-max-free", "Maximum free memory buffers", &mem_max_free);
        callback("mem-initial", "Initial free memory buffers", &mem_initial);
        callback("ring", "Use ringbuffer instead of callbacks", &ring);
        callback("memcpy-nt", "Use non-temporal memcpy", &memcpy_nt);
#if SPEAD2_USE_IBV
        callback("ibv", "Use ibverbs", &ibv);
        callback("ibv-vector", "Interrupt vector (-1 for polled)", &ibv_comp_vector);
        callback("ibv-max-poll", "Maximum number of times to poll in a row", &ibv_max_poll);
#endif
    }

    stream_config make_stream_config(const protocol_options &protocol) const;
    ring_stream_config make_ring_stream_config() const;

    void add_readers(
        spead2::recv::stream &stream,
        const std::vector<std::string> &endpoints,
        const protocol_options &protocol,
        bool allow_pcap) const;
};

} // namespace recv

namespace send
{

struct sender_options
{
    int heap_address_bits = spead2::flavour().get_heap_address_bits();
    std::size_t max_packet_size = spead2::send::stream_config::default_max_packet_size;
    std::string interface_address;
    boost::optional<std::size_t> buffer_size;
    std::size_t burst_size = spead2::send::stream_config::default_burst_size;
    double burst_rate_ratio = spead2::send::stream_config::default_burst_rate_ratio;
    std::size_t max_heaps = spead2::send::stream_config::default_max_heaps;
    bool allow_hw_rate = false;
    double rate = 0.0;
    int ttl = 1;
#if SPEAD2_USE_IBV
    bool ibv = false;
    int ibv_comp_vector = 0;
    int ibv_max_poll = spead2::send::udp_ibv_stream_config::default_max_poll;
#endif

    void notify(const protocol_options &protocol);

    template<typename T>
    void enumerate(T &&callback)
    {
        callback("addr-bits", "Heap address bits", &heap_address_bits);
        callback("packet", "Maximum packet size to send", &max_packet_size);
        callback("bind", "Local address to bind sockets to", &interface_address);
        callback("buffer", "Socket buffer size", &buffer_size);
        callback("burst", "Burst size", &burst_size);
        callback("burst-rate-ratio", "Hard rate limit, relative to --rate", &burst_rate_ratio);
        callback("max-heaps", "Maximum heaps in flight", &max_heaps);
        callback("allow-hw-rate", "Use hardware rate limiting if available", &allow_hw_rate);
        callback("rate", "Transmission rate bound (Gb/s)", &rate);
        callback("ttl", "TTL for multicast target", &ttl);
#if SPEAD2_USE_IBV
        callback("ibv", "Use ibverbs", &ibv);
        callback("ibv-vector", "Interrupt vector (-1 for polled)", &ibv_comp_vector);
        callback("ibv-max-poll", "Maximum number of times to poll in a row", &ibv_max_poll);
#endif
    }

    flavour make_flavour(const protocol_options &protocol) const;
    stream_config make_stream_config() const;

    std::unique_ptr<stream> make_stream(
        boost::asio::io_service &io_service,
        const protocol_options &protocol,
        const std::vector<std::string> &endpoints,
        const std::vector<std::pair<const void *, std::size_t>> &memory_regions) const;
};

} // namespace send

} // namespace spead2

#endif // SPEAD2_CMDLINE_H
