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
 * @file Shared command-line processing for tools
 */

#include <cassert>
#include <iostream>
#include <memory>
#include <functional>
#include <vector>
#include <type_traits>
#include <boost/program_options.hpp>
#include <spead2/common_features.h>
#include <spead2/common_memory_pool.h>
#include <spead2/recv_tcp.h>
#include <spead2/recv_udp.h>
#if SPEAD2_USE_PCAP
# include <spead2/recv_udp_pcap.h>
#endif
#include <spead2/send_tcp.h>
#include <spead2/send_udp.h>
#if SPEAD2_USE_IBV
# include <spead2/recv_udp_ibv.h>
# include <spead2/send_udp_ibv.h>
#endif
#include "spead2_cmdline.h"

namespace po = boost::program_options;
using boost::asio::ip::udp;
using boost::asio::ip::tcp;

namespace spead2
{

template<typename Proto>
std::vector<boost::asio::ip::basic_endpoint<Proto>> parse_endpoints(
    boost::asio::io_service &io_service, const std::vector<std::string> &endpoints, bool passive)
{
    typedef boost::asio::ip::basic_resolver<Proto> resolver_type;
    resolver_type resolver(io_service);
    std::vector<boost::asio::ip::basic_endpoint<Proto>> ans;
    ans.reserve(endpoints.size());
    for (const std::string &dest : endpoints)
    {
        auto colon = dest.rfind(':');
        std::string host, port;
        if (colon == std::string::npos || colon == 0)
        {
            if (!passive)
                throw std::runtime_error("Destination '" + dest + "' does not have the format host:port");
            else
                port = (colon == 0) ? dest.substr(1) : dest;
        }
        else
        {
            host = dest.substr(0, colon);
            port = dest.substr(colon + 1);
        }
        typename resolver_type::query query(
            host, port,
            passive ? resolver_type::query::passive : typename resolver_type::query::flags());
        ans.push_back(*resolver.resolve(query));
    }
    return ans;
}

template<typename Proto>
std::vector<boost::asio::ip::basic_endpoint<Proto>> parse_endpoints(
    const std::vector<std::string> &endpoints, bool passive)
{
    boost::asio::io_service io_service;
    return parse_endpoints<Proto>(io_service, endpoints, passive);
}

// Explicitly instantiate for TCP and UDP
template std::vector<boost::asio::ip::udp::endpoint> parse_endpoints<boost::asio::ip::udp>(
    boost::asio::io_service &io_service, const std::vector<std::string> &endpoints, bool passive);
template std::vector<boost::asio::ip::tcp::endpoint> parse_endpoints<boost::asio::ip::tcp>(
    boost::asio::io_service &io_service, const std::vector<std::string> &endpoints, bool passive);

template std::vector<boost::asio::ip::udp::endpoint> parse_endpoints<boost::asio::ip::udp>(
    const std::vector<std::string> &endpoints, bool passive);
template std::vector<boost::asio::ip::tcp::endpoint> parse_endpoints<boost::asio::ip::tcp>(
    const std::vector<std::string> &endpoints, bool passive);


void protocol_options::notify()
{
    /* No validation required */
}


namespace recv
{

void receiver_options::notify(const protocol_options &protocol)
{
#if SPEAD2_USE_IBV
    if (ibv && interface_address.empty())
        throw po::error("--ibv requires --bind");
    if (protocol.tcp && ibv)
        throw po::error("--ibv and --tcp are incompatible");
#endif

    if (!buffer_size)
    {
        if (protocol.tcp)
            buffer_size = spead2::recv::tcp_reader::default_buffer_size;
#if SPEAD2_USE_IBV
        else if (ibv)
            buffer_size = spead2::recv::udp_ibv_config::default_buffer_size;
#endif
        else
            buffer_size = spead2::recv::udp_reader::default_buffer_size;
    }

    if (!max_packet_size)
    {
        if (protocol.tcp)
            max_packet_size = spead2::recv::tcp_reader::default_max_size;
#if SPEAD2_USE_IBV
        else if (ibv)
            max_packet_size = spead2::recv::udp_ibv_config::default_max_size;
#endif
        else
            max_packet_size = spead2::recv::udp_reader::default_max_size;
    }
}

stream_config receiver_options::make_stream_config(const protocol_options &protocol) const
{
    stream_config config;
    config.set_max_heaps(max_heaps);
    if (mem_pool)
    {
        std::shared_ptr<spead2::memory_pool> pool = std::make_shared<spead2::memory_pool>(
            mem_lower, mem_upper, mem_max_free, mem_initial);
        config.set_memory_allocator(std::move(pool));
    }
    config.set_memcpy(memcpy_nt ? MEMCPY_NONTEMPORAL : MEMCPY_STD);
    config.set_bug_compat(protocol.pyspead ? BUG_COMPAT_PYSPEAD_0_5_2 : 0);
    return config;
}

ring_stream_config receiver_options::make_ring_stream_config() const
{
    assert(ring);
    return ring_stream_config().set_heaps(ring_heaps);
}

void receiver_options::add_readers(
    spead2::recv::stream &stream,
    const std::vector<std::string> &endpoints,
    const protocol_options &protocol,
    bool allow_pcap) const
{
#if SPEAD2_USE_IBV
    std::vector<udp::endpoint> ibv_endpoints;
#endif
    for (const std::string &endpoint : endpoints)
    {
        std::string host = "";
        std::string port;
        auto colon = endpoint.rfind(':');
        if (colon != std::string::npos)
        {
            host = endpoint.substr(0, colon);
            port = endpoint.substr(colon + 1);
        }
        else
            port = endpoint;

        bool is_pcap = false;
        if (allow_pcap)
        {
            // Check whether this looks like <host>:<port> or <port>. If not,
            // guess that it is a pcap file.
            std::string port;
            auto colon = endpoint.rfind(':');
            if (colon != std::string::npos)
                port = endpoint.substr(colon + 1);
            else
                port = endpoint;
            try
            {
                boost::lexical_cast<std::uint16_t>(port);
            }
            catch (boost::bad_lexical_cast &)
            {
                is_pcap = true;
            }
        }

        if (is_pcap)
        {
#if SPEAD2_USE_PCAP
            stream.emplace_reader<udp_pcap_file_reader>(endpoint);
#else
            throw std::runtime_error("spead2 was compiled without pcap support");
#endif
        }
        else if (protocol.tcp)
        {
            tcp::endpoint ep = parse_endpoint<tcp>(stream.get_io_service(), endpoint, true);
            stream.emplace_reader<spead2::recv::tcp_reader>(ep, *max_packet_size, *buffer_size);
        }
        else
        {
            udp::endpoint ep = parse_endpoint<udp>(stream.get_io_service(), endpoint, true);
#if SPEAD2_USE_IBV
            if (ibv)
            {
                ibv_endpoints.push_back(ep);
            }
            else
#endif
            if (ep.address().is_v4() && !interface_address.empty())
            {
                stream.emplace_reader<spead2::recv::udp_reader>(
                    ep, *max_packet_size, *buffer_size,
                    boost::asio::ip::address_v4::from_string(interface_address));
            }
            else
            {
                if (!interface_address.empty())
                    std::cerr << "--bind is not implemented for IPv6\n";
                stream.emplace_reader<spead2::recv::udp_reader>(ep, *max_packet_size, *buffer_size);
            }
        }
    }
#if SPEAD2_USE_IBV
    if (!ibv_endpoints.empty())
    {
        boost::asio::ip::address interface_address = boost::asio::ip::address::from_string(this->interface_address);
        stream.emplace_reader<spead2::recv::udp_ibv_reader>(
            udp_ibv_config()
                .set_endpoints(ibv_endpoints)
                .set_interface_address(interface_address)
                .set_max_size(*max_packet_size)
                .set_buffer_size(*buffer_size)
                .set_comp_vector(ibv_comp_vector)
                .set_max_poll(ibv_max_poll));
    }
#endif
}

} // namespace recv


namespace send
{

std::ostream &operator<<(std::ostream &o, rate_method method)
{
    switch (method)
    {
    case rate_method::SW: return o << "SW";
    case rate_method::HW: return o << "HW";
    case rate_method::AUTO: return o << "AUTO";
    }
    return o;  // unreachable
}

std::istream &operator>>(std::istream &i, rate_method &method)
{
    std::string name;
    if (i >> name)
    {
        if (name == "SW" || name == "sw")
            method = rate_method::SW;
        else if (name == "HW" || name == "hw")
            method = rate_method::HW;
        else if (name == "AUTO" || name == "auto")
            method = rate_method::AUTO;
        else
            i.setstate(std::ios::failbit | std::ios::badbit);
    }
    return i;
}

void sender_options::notify(const protocol_options &protocol)
{
#if SPEAD2_USE_IBV
    if (protocol.tcp && ibv)
        throw po::error("--tcp and --ibv cannot be used together");
    if (ibv && interface_address.empty())
        throw po::error("--ibv requires --bind");
#endif

    if (!buffer_size)
    {
        if (protocol.tcp)
            buffer_size = tcp_stream::default_buffer_size;
#if SPEAD2_USE_IBV
        else if (ibv)
            buffer_size = udp_ibv_config::default_buffer_size;
#endif
        else
            buffer_size = udp_stream::default_buffer_size;
    }
}

flavour sender_options::make_flavour(const protocol_options &protocol) const
{
    return spead2::flavour(spead2::maximum_version, CHAR_BIT * sizeof(item_pointer_t),
                           heap_address_bits,
                           protocol.pyspead ? BUG_COMPAT_PYSPEAD_0_5_2 : 0);
}

stream_config sender_options::make_stream_config() const
{
    return stream_config()
        .set_max_packet_size(max_packet_size)
        .set_rate(rate * (1e9 / 8))
        .set_burst_size(burst_size)
        .set_burst_rate_ratio(burst_rate_ratio)
        .set_max_heaps(max_heaps)
        .set_rate_method(method);
}

std::unique_ptr<stream> sender_options::make_stream(
    boost::asio::io_service &io_service,
    const protocol_options &protocol,
    const std::vector<std::string> &endpoints,
    const std::vector<std::pair<const void *, std::size_t>> &memory_regions) const
{
    std::unique_ptr<stream> stream;
    stream_config config = make_stream_config();
    boost::asio::ip::address interface_address;
    if (!this->interface_address.empty())
        interface_address = boost::asio::ip::address::from_string(this->interface_address);
    if (protocol.tcp)
    {
        auto ep = parse_endpoints<boost::asio::ip::tcp>(io_service, endpoints, false);
        std::promise<void> promise;
        auto connect_handler = [&promise] (const boost::system::error_code &e) {
            if (e)
                promise.set_exception(std::make_exception_ptr(boost::system::system_error(e)));
            else
                promise.set_value();
        };
        stream.reset(new tcp_stream(
            io_service, connect_handler, ep, config,
            *buffer_size, interface_address));
        promise.get_future().get();
    }
    else
    {
        auto ep = parse_endpoints<boost::asio::ip::udp>(io_service, endpoints, false);
#if SPEAD2_USE_IBV
        if (ibv)
        {
            udp_ibv_config ibv_config;
            ibv_config
                .set_endpoints(ep)
                .set_interface_address(interface_address)
                .set_memory_regions(memory_regions)
                .set_ttl(ttl)
                .set_comp_vector(ibv_comp_vector)
                .set_max_poll(ibv_max_poll)
                .set_buffer_size(*buffer_size);
            stream.reset(new udp_ibv_stream(io_service, config, ibv_config));
        }
        else
#endif // SPEAD2_USE_IBV
        {
            if (ep[0].address().is_multicast())
            {
                if (ep[0].address().is_v4())
                    stream.reset(new udp_stream(
                        io_service, ep, config, *buffer_size,
                        ttl, interface_address));
                else
                {
                    if (!this->interface_address.empty())
                        std::cerr << "--bind is not yet supported for IPv6 multicast, ignoring\n";
                    stream.reset(new udp_stream(
                        io_service, ep, config, *buffer_size, ttl));
                }
            }
            else
            {
                stream.reset(new udp_stream(
                    io_service, ep, config, *buffer_size, interface_address));
            }
        }
    }
    return stream;
}

} // namespace send

} // namespace spead2
