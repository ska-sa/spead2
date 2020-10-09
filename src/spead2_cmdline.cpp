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

namespace spead2
{

template<typename T>
class make_value_semantic
{
public:
    po::typed_value<T> *operator()(const T &def) const
    {
        return po::value<T>()->default_value(def);
    }

    po::typed_value<T> *operator()(const T &def, T *out) const
    {
        return po::value<T>(out)->default_value(def);
    }
};

template<>
class make_value_semantic<bool>
{
public:
    po::typed_value<bool> *operator()(bool def) const
    {
        assert(!def);
        return po::bool_switch();
    }

    po::typed_value<bool> *operator()(bool def, bool *out) const
    {
        assert(!def);
        return po::bool_switch(out);
    }
};

template<typename T>
class option_adder
{
private:
    po::options_description &desc;
    T &value;
    std::map<std::string, std::string> name_map;

public:
    option_adder(po::options_description &desc, T &value,
                 const std::map<std::string, std::string> &name_map)
        : desc(desc), value(value), name_map(name_map)
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

    template<typename Getter, typename Setter>
    void add(const std::string &name, const std::string &description,
             const Getter &getter, const Setter &setter) const
    {
        using namespace std::placeholders;
        // Using std::bind allows pointer-to-member-functions
        auto def = std::bind(getter, value)();
        add(name, description,
            make_value_semantic<decltype(def)>()(def)->notifier(std::bind(setter, value, _1)));
    }

    template<typename U>
    void add(const std::string &name, const std::string &description, U (T::*data))
    {
        add(name, description,
            make_value_semantic<U>()(value.*data, &(value.*data)));
    }
};

po::options_description protocol_config_options(
    protocol_config &config, const std::map<std::string, std::string> &name_map)
{
    po::options_description desc;
    option_adder<protocol_config> adder(desc, config, name_map);
    adder.add("tcp", "Use TCP instead of UDP", &protocol_config::tcp);
    adder.add("pyspead", "Be bug-compatible with PySPEAD", &protocol_config::pyspead);
    return desc;
}


namespace recv
{

po::options_description receiver_config_options(
    receiver_config &config, const std::map<std::string, std::string> &name_map)
{
    po::options_description desc;
    option_adder<receiver_config> adder(desc, config, name_map);
    adder.add("bind", "Interface address", &receiver_config::interface_address);
    adder.add("packet", "Maximum packet size to accept", &receiver_config::max_packet);
    // No default, because the default depends on the protocol
    adder.add("buffer", "Socket buffer size", po::value(&config.buffer));
    adder.add("concurrent-heaps", "Maximum number of in-flight heaps", &receiver_config::max_heaps);
    adder.add("ring-heaps", "Ring buffer capacity in heaps", &receiver_config::ring_heaps);
    adder.add("mem-pool", "Use a memory pool", &receiver_config::mem_pool);
    adder.add("mem-lower", "Minimum allocation which will use the memory pool", &receiver_config::mem_lower);
    adder.add("mem-upper", "Maximum allocation which will use the memory pool", &receiver_config::mem_upper);
    adder.add("mem-max-free", "Maximum free memory buffers", &receiver_config::mem_max_free);
    adder.add("mem-initial", "Initial free memory buffers", &receiver_config::mem_initial);
    adder.add("ring", "Use ringbuffer instead of callbacks", &receiver_config::ring);
    adder.add("memcpy-nt", "Use non-temporal memcpy", &receiver_config::memcpy_nt);
#if SPEAD2_USE_IBV
    adder.add("ibv", "Use ibverbs", &receiver_config::opts.ibv);
    adder.add("ibv-vector", "Interrupt vector (-1 for polled)", &receiver_config::ibv_comp_vector);
    adder.add("ibv-max-poll", "Maximum number of times to poll in a row", &receiver_config::ibv_max_poll);
#endif
    return desc;
}

stream_config options_to_stream_config(const protocol_config &protocol,
                                       const receiver_config &receiver)
{
    stream_config config;
    config.set_max_heaps(receiver.max_heaps);
    if (receiver.mem_pool)
    {
        std::shared_ptr<spead2::memory_pool> pool = std::make_shared<spead2::memory_pool>(
            receiver.mem_lower, receiver.mem_upper, receiver.mem_max_free, receiver.mem_initial);
        config.set_memory_allocator(std::move(pool));
    }
    config.set_memory_allocator(...);
    config.set_memcpy(receiver.memcpy_nt ? MEMCPY_NONTEMPORAL : MEMCPY_STD);
    config.set_bug_compat(protocol.pyspead ? BUG_COMPAT_PYSPEAD_0_5_2 : 0);
    return config;
}

void add_readers(
    spead2::recv::stream &stream,
    const std::vector<std::string> &endpoints,
    const protocol_config &protocol,
    const receiver_config &receiver,
    bool allow_pcap)
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
            stream.emplace_reader<spead2::recv::tcp_reader>(endpoint, receiver.max_packet, receiver.buffer);
        }
        else
        {
            udp::resolver resolver(stream.get_io_service());
            udp::resolver::query query(host, port,
                boost::asio::ip::udp::resolver::query::address_configured
                | boost::asio::ip::udp::resolver::query::passive);
            udp::endpoint endpoint = *resolver.resolve(query);
#if SPEAD2_USE_IBV
            if (receiver.ibv)
            {
                ibv_endpoints.push_back(endpoint);
            }
            else
#endif
            if (endpoint.address().is_v4() && !receiver.interface_address.empty())
            {
                stream.emplace_reader<spead2::recv::udp_reader>(
                    endpoint, receiver.max_packet, receiver.buffer,
                    boost::asio::ip::address_v4::from_string(receiver.interface_address));
            }
            else
            {
                if (!receiver.interface_address.empty())
                    std::cerr << "--bind is not implemented for IPv6\n";
                stream.emplace_reader<spead2::recv::udp_reader>(endpoint, receiver.max_packet, receiver.buffer);
            }
        }
    }
#if SPEAD2_USE_IBV
    if (!ibv_endpoints.empty())
    {
        boost::asio::ip::address interface_address = boost::asio::ip::address::from_string(receiver.interface_address);
        stream.emplace_reader<spead2::recv::udp_ibv_reader>(
            ibv_endpoints, interface_address, receiver.max_packet, receiver.buffer,
            receiver.ibv_comp_vector, receiver.ibv_max_poll);
    }
#endif
}

} // namespace recv


namespace send
{

void writer_config::finalize(const protocol_config &protocol)
{
#if SPEAD2_USE_IBV
    if (protocol.tcp && ibv)
        throw po::error("--tcp and --ibv cannot be used together");
    if (ibv && interface_address.is_unspecified())
        throw po::error("--ibv requires --bind");
#endif

    if (!buffer)
    {
        if (protocol.tcp)
            buffer = tcp_stream::default_buffer_size;
#if SPEAD2_USE_IBV
        else if (ibv)
            buffer = udp_ibv_stream_config::default_buffer_size;
#endif
        else
            buffer = udp_stream::default_buffer_size;
    }

    int bug_compat = protocol.pyspead
        ? flavour.get_bug_compat() | BUG_COMPAT_PYSPEAD_0_5_2
        : flavour.get_bug_compat() & ~BUG_COMPAT_PYSPEAD_0_5_2;
    flavour = spead2::flavour(
        flavour.get_version(),
        flavour.get_item_pointer_bits(),
        flavour.get_heap_address_bits(),
        bug_compat);

#if SPEAD2_USE_IBV
    udp_ibv_config.set_ttl(ttl);
    udp_ibv_config.set_buffer_size(*buffer);
    udp_ibv_config.set_interface_address(interface_address);
#endif
}

po::options_description flavour_options(
    spead2::flavour &flavour, const std::map<std::string, std::string> &name_map)
{
    po::options_description desc;
    option_adder<spead2::flavour> adder(desc, flavour, name_map);
    // --pyspead is handled in protocol_config
    adder.add(
        "addr-bits", "Heap address bits",
        &flavour::get_heap_address_bits,
        [](spead2::flavour &f, int addr_bits)
        {
            // flavour is immutable, so we have to replace the whole object.
            f = spead2::flavour(
                f.get_version(),
                f.get_item_pointer_bits(),
                addr_bits,
                f.get_bug_compat());
        });
    return desc;
}

po::options_description stream_config_options(
    stream_config &config, const std::map<std::string, std::string> &name_map)
{
    po::options_description desc;
    option_adder<stream_config> adder(desc, config, name_map);
    adder.add("packet", "Maximum packet size to send",
              &stream_config::get_max_packet_size,
              &stream_config::set_max_packet_size);
    adder.add("burst", "Burst size",
              &stream_config::get_burst_size,
              &stream_config::set_burst_size);
    adder.add("burst-rate-ratio", "Hard rate limit, relative to --rate",
              &stream_config::get_burst_rate_ratio,
              &stream_config::set_burst_rate_ratio);
    adder.add("allow-hw-rate", "Use hardware rate limiting if available",
              &stream_config::get_allow_hw_rate,
              &stream_config::set_allow_hw_rate);
    adder.add("max-heaps", "Maximum heaps in flight",
              &stream_config::get_max_heaps,
              &stream_config::set_max_heaps);
    adder.add("rate", "Transmission rate bound (Gb/s)",
              &stream_config::get_rate,
              &stream_config::set_rate);
    return desc;
}

#if SPEAD2_USE_IBV
po::options_description udp_ibv_stream_config_options(
    udp_ibv_stream_config &config, const std::map<std::string, std::string> &name_map)
{
    po::options_description desc;
    option_adder<udp_ibv_stream_config> adder(desc, config, name_map);

    // interface address and TTL are configured elsewhere
    adder.add("ibv-vector", "Interrupt vector (-1 for polled)",
              &udp_ibv_stream_config::get_comp_vector,
              &udp_ibv_stream_config::set_comp_vector);
    adder.add("ibv-max-poll", "Maximum number of times to poll in a row",
              &udp_ibv_stream_config::get_max_poll,
              &udp_ibv_stream_config::set_max_poll);
    return desc;
}
#endif // SPEAD2_USE_IBV

po::options_description writer_config_options(
    writer_config &config, const std::map<std::string, std::string> &name_map)
{
    po::options_description desc;
    option_adder<writer_config> adder(desc, config, name_map);
    desc.add(flavour_options(config.flavour, name_map));
#if SPEAD2_USE_IBV
    adder.add("ibv", "Use ibverbs", &writer_config::ibv);
    desc.add(udp_ibv_stream_config_options(config.udp_ibv_config, name_map));
#endif
    // We don't set a default for buffer, because it depends on the writer
    adder.add("buffer", "Socket buffer size", po::value(&config.buffer));
    adder.add("ttl", "TTL for multicast target", &writer_config::ttl);
    adder.add(
        "bind", "Local address to bind sockets to",
        [](const writer_config &config)
        {
            return config.interface_address.is_unspecified() ? "" : config.interface_address.to_string();
        },
        [](writer_config &config, const std::string &address)
        {
            if (address.empty())
                config.interface_address = boost::asio::ip::address();
            else
                config.interface_address = boost::asio::ip::address::from_string(address);
        });
    desc.add(stream_config_options(config.config, name_map));
    return desc;
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

std::unique_ptr<stream> make_stream(
    boost::asio::io_service &io_service,
    const protocol_config &protocol, const writer_config &writer,
    const std::vector<std::string> &endpoints,
    const std::vector<std::pair<const void *, std::size_t>> &memory_regions)
{
    std::unique_ptr<stream> stream;
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
            io_service, connect_handler, ep,
            writer.config, *writer.buffer, writer.interface_address));
        promise.get_future().get();
    }
    else
    {
        auto ep = get_endpoints<boost::asio::ip::udp>(io_service, endpoints);
#if SPEAD2_USE_IBV
        if (writer.ibv)
        {
            udp_ibv_stream_config udp_ibv_config = writer.udp_ibv_config;
            udp_ibv_config
                .set_endpoints(ep)
                .set_memory_regions(memory_regions);
            stream.reset(new udp_ibv_stream(
                    io_service, writer.config, udp_ibv_config));
        }
        else
#endif // SPEAD2_USE_IBV
        {
            if (ep[0].address().is_multicast())
            {
                if (ep[0].address().is_v4())
                    stream.reset(new udp_stream(
                        io_service, ep, writer.config, *writer.buffer,
                        writer.ttl, writer.interface_address));
                else
                {
                    if (!writer.interface_address.is_unspecified())
                        std::cerr << "--bind is not yet supported for IPv6 multicast, ignoring\n";
                    stream.reset(new udp_stream(
                        io_service, ep, writer.config, *writer.buffer,
                        writer.ttl));
                }
            }
            else
            {
                stream.reset(new udp_stream(
                    io_service, ep, writer.config, *writer.buffer, writer.interface_address));
            }
        }
    }
    return stream;
}

} // namespace send

} // namespace spead2
