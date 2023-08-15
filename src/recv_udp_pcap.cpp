/* Copyright 2016-2017, 2019, 2023 National Research Foundation (SARAO)
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

#include <spead2/common_features.h>
#if SPEAD2_USE_PCAP
#include <cassert>
#include <cstdint>
#include <string>
#include <functional>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp_base.h>
#include <spead2/recv_udp_pcap.h>
#include <spead2/common_raw_packet.h>
#include <spead2/common_logging.h>

// These are defined in pcap/dlt.h for libpcap >= 1.8.0, but in pcap/bfp.h otherwise
// We define them here to avoid having to guess which file to include
#ifndef DLT_EN10MB
#define DLT_EN10MB 1
#endif
#ifndef DLT_LINUX_SLL
#define DLT_LINUX_SLL 113
#endif

namespace spead2::recv
{

void udp_pcap_file_reader::run(handler_context ctx, stream_base::add_packet_state &state)
{
    const int BATCH = 64;  // maximum number of packets to process in one go

    for (int pass = 0; pass < BATCH; pass++)
    {
        if (state.is_stopped())
            break;
        struct pcap_pkthdr *h;
        const u_char *pkt_data;
        int status = pcap_next_ex(handle, &h, &pkt_data);
        switch (status)
        {
        case 1:
            // Successful read
            if (h->caplen < h->len)
            {
                log_warning("Packet was truncated (%d < %d)", h->caplen, h->len);
            }
            else
            {
                try
                {
                    void *bytes = const_cast<void *>((const void *) pkt_data);
                    packet_buffer payload = udp_from_frame(bytes, h->len);
                    process_one_packet(state, payload.data(), payload.size(), payload.size());
                }
                catch (packet_type_error &e)
                {
                    log_warning(e.what());
                }
                catch (std::length_error &e)
                {
                    log_warning(e.what());
                }
            }
            break;
        case -1:
            log_warning("Error reading packet: %s", pcap_geterr(handle));
            break;
        case -2:
            // End of file
            state.stop();
            break;
        }
    }
    // Run ourselves again
    if (!state.is_stopped())
    {
        using namespace std::placeholders;
        boost::asio::post(get_io_service(), bind_handler(std::move(ctx), std::bind(&udp_pcap_file_reader::run, this, _1, _2)));
    }
}

udp_pcap_file_reader::udp_pcap_file_reader(stream &owner, const std::string &filename, const std::string &user_filter)
    : udp_reader_base(owner)
{
    // Open the file
    char errbuf[PCAP_ERRBUF_SIZE];
    handle = pcap_open_offline(filename.c_str(), errbuf);
    if (!handle)
        throw std::runtime_error(errbuf);
    // Set a filter to ensure that we only get UDP4 packets with no fragmentation
    bpf_program filter;
    std::string filter_expression = "ip proto \\udp and ip[6:2] & 0x3fff = 0";
    if (!user_filter.empty())
        filter_expression += " and (" + user_filter + ')';
    if (pcap_compile(handle, &filter,
                     filter_expression.c_str(),
                     1, PCAP_NETMASK_UNKNOWN) != 0)
        throw std::runtime_error(pcap_geterr(handle));
    if (pcap_setfilter(handle, &filter) != 0)
    {
        std::runtime_error error(pcap_geterr(handle));
        pcap_freecode(&filter);
        throw error;
    }
    pcap_freecode(&filter);
    // The link type used to record this file
    auto linktype = pcap_datalink(handle);
    assert(linktype != PCAP_ERROR_NOT_ACTIVATED);
    if (linktype != DLT_EN10MB && linktype != DLT_LINUX_SLL)
        throw packet_type_error("pcap linktype is neither ethernet nor linux sll");
    udp_from_frame = (linktype == DLT_EN10MB) ? udp_from_ethernet : udp_from_linux_sll;

    // Process the file
    using namespace std::placeholders;
    boost::asio::post(get_io_service(), bind_handler(std::bind(&udp_pcap_file_reader::run, this, _1, _2)));
}

udp_pcap_file_reader::~udp_pcap_file_reader()
{
    if (handle)
        pcap_close(handle);
}

bool udp_pcap_file_reader::lossy() const
{
    return false;
}

} // namespace spead2::recv

#endif // SPEAD2_USE_PCAP
