/* Copyright 2016-2017, 2019 SKA South Africa
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
#include <cstdint>
#include <string>
#include <spead2/recv_reader.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp_base.h>
#include <spead2/recv_udp_pcap.h>
#include <spead2/common_raw_packet.h>
#include <spead2/common_logging.h>

namespace spead2
{
namespace recv
{

void udp_pcap_file_reader::run()
{
    const int BATCH = 64;  // maximum number of packets to process in one go

    spead2::recv::stream_base::add_packet_state state(get_stream_base());
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
                    packet_buffer payload = udp_from_ethernet(bytes, h->len);
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
        get_io_service().post([this] { run(); });
    else
        stopped();
}

udp_pcap_file_reader::udp_pcap_file_reader(stream &owner, const std::string &filename)
    : udp_reader_base(owner)
{
    // Open the file
    char errbuf[PCAP_ERRBUF_SIZE];
    handle = pcap_open_offline(filename.c_str(), errbuf);
    if (!handle)
        throw std::runtime_error(errbuf);
    // Set a filter to ensure that we only get UDP4 packets with no fragmentation
    bpf_program filter;
    if (pcap_compile(handle, &filter,
                     "ip proto \\udp and ip[6:2] & 0x3fff = 0",
                     1, PCAP_NETMASK_UNKNOWN) != 0)
        throw std::runtime_error(pcap_geterr(handle));
    if (pcap_setfilter(handle, &filter) != 0)
    {
        std::runtime_error error(pcap_geterr(handle));
        pcap_freecode(&filter);
        throw error;
    }
    pcap_freecode(&filter);

    // Process the file
    get_io_service().post([this] { run(); });
}

udp_pcap_file_reader::~udp_pcap_file_reader()
{
    if (handle)
        pcap_close(handle);
}

void udp_pcap_file_reader::stop()
{
}

bool udp_pcap_file_reader::lossy() const
{
    return false;
}

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_PCAP
