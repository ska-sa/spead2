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
    boost::optional<std::size_t> max_packet;
    std::string interface_address;
#if SPEAD2_USE_IBV
    bool ibv = false;
    int ibv_comp_vector = 0;
    int ibv_max_poll = spead2::recv::udp_ibv_reader::default_max_poll;
#endif

    void notify();

    boost::program_options::options_description make_options(
        const std::map<std::string, std::string> &name_map = {});

    stream_config make_stream_config() const;
    ring_stream_config make_ring_stream_config() const;

    void add_readers(
        spead2::recv::stream &stream,
        const std::vector<std::string> &endpoints,
        const protocol_options &protocol,
        bool allow_pcap);
};

} // namespace recv

namespace send
{

struct writer_config
{
    spead2::flavour flavour;
    stream_config config;
#if SPEAD2_USE_IBV
    bool ibv = false;
    udp_ibv_stream_config udp_ibv_config;
#endif
    int ttl = 1;
    boost::optional<std::size_t> buffer;
    boost::asio::ip::address interface_address;

    void finalize(const protocol_options &protocol);
};

boost::program_options::options_description flavour_options(
    spead2::flavour &flavour, const std::map<std::string, std::string> &name_map = {});

boost::program_options::options_description writer_config_options(
    writer_config &config, const std::map<std::string, std::string> &name_map = {});

std::unique_ptr<stream> make_stream(
    boost::asio::io_service &io_service,
    const protocol_options &protocol, const writer_config &writer,
    const std::vector<std::string> &endpoints,
    const std::vector<std::pair<const void *, std::size_t>> &memory_regions);

} // namespace send

} // namespace spead2

#endif // SPEAD2_CMDLINE_H
