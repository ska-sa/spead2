/* Copyright 2016-2017, 2023 National Research Foundation (SARAO)
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

#ifndef SPEAD2_RECV_UDP_PCAP
#define SPEAD2_RECV_UDP_PCAP

#include <spead2/common_features.h>
#if SPEAD2_USE_PCAP

#include <cstdint>
#include <string>
#include <pcap/pcap.h>
#include <spead2/common_raw_packet.h>
#include <spead2/recv_udp_base.h>
#include <spead2/recv_stream.h>

namespace spead2::recv
{

/**
 * Reader class that feeds data from a pcap file to a stream.
 *
 * An optional filter selects a subset of the packets in the capture file.
 */
class udp_pcap_file_reader : public udp_reader_base
{
private:
    using udp_unpacker = packet_buffer (*)(void *, size_t);
    pcap_t *handle;
    udp_unpacker udp_from_frame;

    void run(handler_context ctx, stream_base::add_packet_state &state);

public:
    /**
     * Constructor
     *
     * @param owner Owning stream
     * @param filename Filename of the capture file
     * @param filter Filter to apply to packets from the capture file
     *
     * @throws std::runtime_error if @a filename could not read
     */
    udp_pcap_file_reader(stream &owner, const std::string &filename, const std::string &filter = "");
    virtual ~udp_pcap_file_reader();

    virtual void start() override;
    virtual bool lossy() const override;
};

} // namespace spead2::recv

#endif // SPEAD2_USE_PCAP
#endif // SPEAD2_RECV_UDP_PCAP
