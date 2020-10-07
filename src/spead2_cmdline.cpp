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
#include <spead2/send_tcp.h>
#include <spead2/send_udp.h>
#if SPEAD2_USE_IBV
# include <spead2/send_udp_ibv.h>
#endif
#include "spead2_cmdline.h"

namespace po = boost::program_options;

namespace spead2
{

static std::string apply_prefix(const std::string &prefix, const std::string &option)
{
    if (!prefix.empty())
        return prefix + "-" + option;
    else
        return option;
}


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
    std::string prefix;

public:
    option_adder(po::options_description &desc, T &value, const std::string &prefix)
        : desc(desc), prefix(prefix), value(value)
    {
    }

    template<typename Getter, typename Setter>
    void add(const std::string &name, const std::string &description,
             const Getter &getter, const Setter &setter) const
    {
        using namespace std::placeholders;
        auto def = std::bind(getter, value)();
        desc.add_options()
            (apply_prefix(prefix, name).c_str(),
             make_value_semantic<decltype(def)>()(def)->notifier(std::bind(setter, value, _1)),
             description.c_str());
    }

    template<typename U>
    void add(const std::string &name, const std::string &description, U (T::*data))
    {
        desc.add_options()
            (apply_prefix(prefix, name).c_str(),
             make_value_semantic<U>()(value.*data, &(value.*data)),
             description.c_str());
    }

    void add(const std::string &name, const std::string &description, po::value_semantic *value)
    {
        desc.add_options()
            (apply_prefix(prefix, name).c_str(),
             value,
             description.c_str());
    }
};

po::options_description protocol_config_options(
    protocol_config &config, const std::string &prefix)
{
    po::options_description desc;
    option_adder<protocol_config> adder(desc, config, prefix);
    adder.add("tcp", "Use TCP instead of UDP", &protocol_config::tcp);
    adder.add("pyspead", "Be bug-compatible with PySPEAD", &protocol_config::pyspead);
    return desc;
}


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
    spead2::flavour &flavour, const std::string &prefix)
{
    po::options_description desc;
    option_adder<spead2::flavour> adder(desc, flavour, prefix);
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
    stream_config &config, bool include_rate, const std::string &prefix)
{
    po::options_description desc;
    option_adder<stream_config> adder(desc, config, prefix);
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
    if (include_rate)
    {
        adder.add("rate", "Transmission rate bound (Gb/s)",
                  &stream_config::get_rate,
                  &stream_config::set_rate);
    }
    return desc;
}

#if SPEAD2_USE_IBV
po::options_description udp_ibv_stream_config_options(
    udp_ibv_stream_config &config, const std::string &prefix)
{
    po::options_description desc;
    option_adder<udp_ibv_stream_config> adder(desc, config, prefix);

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
    writer_config &config, bool include_rate, const std::string &prefix)
{
    po::options_description desc;
    option_adder<writer_config> adder(desc, config, prefix);
    desc.add(flavour_options(config.flavour, prefix));
#if SPEAD2_USE_IBV
    adder.add("ibv", "Use ibverbs", &writer_config::ibv);
    desc.add(udp_ibv_stream_config_options(config.udp_ibv_config, prefix));
#endif
    // We don't set a default for buffer, because it depends on the writer
    adder.add("buffer", "Socket buffer size", po::value<std::size_t>());
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
    desc.add(stream_config_options(config.config, include_rate, prefix));
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
