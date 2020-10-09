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

template<typename T>
class make_value_semantic
{
public:
    po::typed_value<T> *operator()(T *out) const
    {
        return po::value<T>(out)->default_value(*out);
    }
};

template<typename T>
class make_value_semantic<boost::optional<T>>
{
public:
    po::typed_value<boost::optional<T>> *operator()(boost::optional<T> *out) const
    {
        return po::value<boost::optional<T>>(out);
    }
};

template<>
class make_value_semantic<bool>
{
public:
    po::typed_value<bool> *operator()(bool *out) const
    {
        assert(!*out);
        return po::bool_switch(out);
    }
};

class option_adder
{
private:
    po::options_description &desc;
    std::map<std::string, std::string> name_map;

public:
    option_adder(po::options_description &desc, const std::map<std::string, std::string> &name_map)
        : desc(desc), name_map(name_map)
    {
    }

    void add(const std::string &name, const std::string &description, po::value_semantic *value) const
    {
        auto it = name_map.find(name);
        std::string new_name = it == name_map.end() ? name : it->second;
        if (!new_name.empty())
        {
            desc.add_options()
                (new_name.c_str(), value, description.c_str());
        }
        else
            delete value;
    }

    template<typename U>
    typename std::enable_if<!std::is_convertible<U *, po::value_semantic *>::value>::type
    add(const std::string &name, const std::string &description, U *value)
    {
        add(name, description, make_value_semantic<U>()(value));
    }
};

void protocol_options::notify()
{
    /* No validation required */
}

po::options_description protocol_options::make_options(
    const std::map<std::string, std::string> &name_map)
{
    po::options_description desc;
    option_adder adder(desc, name_map);
    adder.add("tcp", "Use TCP instead of UDP", &tcp);
    adder.add("pyspead", "Be bug-compatible with PySPEAD", &pyspead);
    return desc;
}


namespace recv
{

po::options_description receiver_options::make_options(
    const std::map<std::string, std::string> &name_map)
{
    po::options_description desc;
    option_adder adder(desc, name_map);
    adder.add("bind", "Interface address", &interface_address);
    adder.add("packet", "Maximum packet size to accept", &max_packet_size);
    adder.add("buffer", "Socket buffer size", &buffer_size);
    adder.add("concurrent-heaps", "Maximum number of in-flight heaps", &max_heaps);
    adder.add("ring-heaps", "Ring buffer capacity in heaps", &ring_heaps);
    adder.add("mem-pool", "Use a memory pool", &mem_pool);
    adder.add("mem-lower", "Minimum allocation which will use the memory pool", &mem_lower);
    adder.add("mem-upper", "Maximum allocation which will use the memory pool", &mem_upper);
    adder.add("mem-max-free", "Maximum free memory buffers", &mem_max_free);
    adder.add("mem-initial", "Initial free memory buffers", &mem_initial);
    adder.add("ring", "Use ringbuffer instead of callbacks", &ring);
    adder.add("memcpy-nt", "Use non-temporal memcpy", &memcpy_nt);
#if SPEAD2_USE_IBV
    adder.add("ibv", "Use ibverbs", &ibv);
    adder.add("ibv-vector", "Interrupt vector (-1 for polled)", &ibv_comp_vector);
    adder.add("ibv-max-poll", "Maximum number of times to poll in a row", &ibv_max_poll);
#endif
    return desc;
}

void receiver_options::notify(const protocol_options &protocol)
{
#if SPEAD2_USE_IBV
    if (ibv && interface_address.empty())
        throw po::error("--ibv requires --bind");
    if (protocol.tcp && ibv)
        throw po::error("--ibv and --tcp are incompatible");
#endif
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
            tcp::resolver resolver(stream.get_io_service());
            tcp::resolver::query query(host, port, tcp::resolver::query::address_configured);
            tcp::endpoint endpoint = *resolver.resolve(query);
            std::size_t buffer_size = this->buffer_size.value_or(spead2::recv::tcp_reader::default_buffer_size);
            std::size_t max_packet_size = this->max_packet_size.value_or(spead2::recv::tcp_reader::default_max_size);
            stream.emplace_reader<spead2::recv::tcp_reader>(endpoint, max_packet_size, buffer_size);
        }
        else
        {
            udp::resolver resolver(stream.get_io_service());
            udp::resolver::query query(host, port,
                boost::asio::ip::udp::resolver::query::address_configured
                | boost::asio::ip::udp::resolver::query::passive);
            udp::endpoint endpoint = *resolver.resolve(query);
            std::size_t buffer_size = this->buffer_size.value_or(spead2::recv::udp_reader::default_buffer_size);
            std::size_t max_packet_size = this->max_packet_size.value_or(spead2::recv::udp_reader::default_max_size);
#if SPEAD2_USE_IBV
            if (ibv)
            {
                ibv_endpoints.push_back(endpoint);
            }
            else
#endif
            if (endpoint.address().is_v4() && !interface_address.empty())
            {
                stream.emplace_reader<spead2::recv::udp_reader>(
                    endpoint, max_packet_size, buffer_size,
                    boost::asio::ip::address_v4::from_string(interface_address));
            }
            else
            {
                if (!interface_address.empty())
                    std::cerr << "--bind is not implemented for IPv6\n";
                stream.emplace_reader<spead2::recv::udp_reader>(endpoint, max_packet_size, buffer_size);
            }
        }
    }
#if SPEAD2_USE_IBV
    if (!ibv_endpoints.empty())
    {
        boost::asio::ip::address interface_address = boost::asio::ip::address::from_string(this->interface_address);
        std::size_t buffer_size = this->buffer_size.value_or(spead2::recv::udp_ibv_reader::default_buffer_size);
        std::size_t max_packet_size = this->max_packet_size.value_or(spead2::recv::udp_ibv_reader::default_max_size);
        stream.emplace_reader<spead2::recv::udp_ibv_reader>(
            ibv_endpoints, interface_address, max_packet_size, buffer_size,
            ibv_comp_vector, ibv_max_poll);
    }
#endif
}

} // namespace recv


namespace send
{

void sender_options::notify(const protocol_options &protocol)
{
#if SPEAD2_USE_IBV
    if (protocol.tcp && ibv)
        throw po::error("--tcp and --ibv cannot be used together");
    if (ibv && interface_address.empty())
        throw po::error("--ibv requires --bind");
#endif
}

po::options_description sender_options::make_options(
    const std::map<std::string, std::string> &name_map)
{
    po::options_description desc;
    option_adder adder(desc, name_map);
    adder.add("addr-bits", "Heap address bits", &heap_address_bits);
    adder.add("packet", "Maximum packet size to send", &max_packet_size);
    adder.add("bind", "Local address to bind sockets to", &interface_address);
    adder.add("buffer", "Socket buffer size", &buffer_size);
    adder.add("burst", "Burst size", &burst_size);
    adder.add("burst-rate-ratio", "Hard rate limit, relative to --rate", &burst_rate_ratio);
    adder.add("max-heaps", "Maximum heaps in flight", &max_heaps);
    adder.add("allow-hw-rate", "Use hardware rate limiting if available", &allow_hw_rate);
    adder.add("rate", "Transmission rate bound (Gb/s)", &rate);
    adder.add("ttl", "TTL for multicast target", &ttl);
#if SPEAD2_USE_IBV
    adder.add("ibv", "Use ibverbs", &ibv);
    adder.add("ibv-vector", "Interrupt vector (-1 for polled)", &ibv_comp_vector);
    adder.add("ibv-max-poll", "Maximum number of times to poll in a row", &ibv_max_poll);
#endif
    return desc;
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
        .set_allow_hw_rate(allow_hw_rate);
}

template<typename Proto>
static std::vector<boost::asio::ip::basic_endpoint<Proto>> get_endpoints(
    boost::asio::io_service &io_service, const std::vector<std::string> &endpoints)
{
    typedef boost::asio::ip::basic_resolver<Proto> resolver_type;
    resolver_type resolver(io_service);
    std::vector<boost::asio::ip::basic_endpoint<Proto>> ans;
    ans.reserve(endpoints.size());
    for (const std::string &dest : endpoints)
    {
        auto colon = dest.rfind(':');
        if (colon == std::string::npos)
            throw std::runtime_error("Destination '" + dest + "' does not have the format host:port");
        typename resolver_type::query query(dest.substr(0, colon), dest.substr(colon + 1));
        ans.push_back(*resolver.resolve(query));
    }
    return ans;
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
        auto ep = get_endpoints<boost::asio::ip::tcp>(io_service, endpoints);
        std::promise<void> promise;
        auto connect_handler = [&promise] (const boost::system::error_code &e) {
            if (e)
                promise.set_exception(std::make_exception_ptr(boost::system::system_error(e)));
            else
                promise.set_value();
        };
        stream.reset(new tcp_stream(
            io_service, connect_handler, ep, config,
            buffer_size.value_or(tcp_stream::default_buffer_size), interface_address));
        promise.get_future().get();
    }
    else
    {
        auto ep = get_endpoints<boost::asio::ip::udp>(io_service, endpoints);
#if SPEAD2_USE_IBV
        if (ibv)
        {
            udp_ibv_stream_config udp_ibv_config;
            udp_ibv_config
                .set_endpoints(ep)
                .set_interface_address(interface_address)
                .set_memory_regions(memory_regions)
                .set_ttl(ttl)
                .set_comp_vector(ibv_comp_vector)
                .set_max_poll(ibv_max_poll);
            if (buffer_size)
                udp_ibv_config.set_buffer_size(*buffer_size);
            stream.reset(new udp_ibv_stream(io_service, config, udp_ibv_config));
        }
        else
#endif // SPEAD2_USE_IBV
        {
            std::size_t buffer_size = this->buffer_size.value_or(udp_stream::default_buffer_size);
            if (ep[0].address().is_multicast())
            {
                if (ep[0].address().is_v4())
                    stream.reset(new udp_stream(
                        io_service, ep, config, buffer_size,
                        ttl, interface_address));
                else
                {
                    if (!this->interface_address.empty())
                        std::cerr << "--bind is not yet supported for IPv6 multicast, ignoring\n";
                    stream.reset(new udp_stream(
                        io_service, ep, config, buffer_size, ttl));
                }
            }
            else
            {
                stream.reset(new udp_stream(
                    io_service, ep, config, buffer_size, interface_address));
            }
        }
    }
    return stream;
}

} // namespace send

} // namespace spead2
