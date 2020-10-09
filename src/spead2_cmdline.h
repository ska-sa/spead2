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

#ifndef SPEAD2_CMDLINE_H
#define SPEAD2_CMDLINE_H

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>
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

struct protocol_options
{
    bool tcp = false;
    bool pyspead = false;

    void notify();

    boost::program_options::options_description make_options(
        const std::map<std::string, std::string> &name_map = {});
};

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

    boost::program_options::options_description make_options(
        const std::map<std::string, std::string> &name_map = {});

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

    boost::program_options::options_description make_options(
        const std::map<std::string, std::string> &name_map = {});

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
