/* Copyright 2015 SKA South Africa
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

#if SPEAD2_USE_NETMAP

#include <cstdint>
#include <boost/asio.hpp>
#include <cerrno>
#include <system_error>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <net/ethernet.h>
#include "recv_reader.h"
#include "recv_netmap.h"
#include "common_logging.h"

namespace spead2
{
namespace recv
{

namespace detail
{

void nm_desc_destructor::operator()(nm_desc *d) const
{
    // We wrap the fd in an asio handle, which takes care of closing it.
    // To prevent nm_close from closing it too, we nullify it here.
    d->fd = -1;
    int status = nm_close(d);
    if (status != 0)
    {
        std::error_code code(status, std::system_category());
        log_warning("Failed to close the netmap fd: %1% (%2%)", code.value(), code.message());
    }
}

} // namespace detail

netmap_udp_reader::netmap_udp_reader(stream &owner, const std::string &device, std::uint16_t port)
    : reader(owner), handle(get_io_service()),
    desc(nm_open(("netmap:" + device + "*").c_str(), NULL, 0, NULL)),
    port_be(htons(port))
{
    if (!desc)
        throw std::system_error(errno, std::system_category());
    handle.assign(desc->fd);
    enqueue_receive();
}

void netmap_udp_reader::packet_handler(const boost::system::error_code &error)
{
    struct header
    {
        ether_header eth;
        iphdr ip;
        udphdr udp;
    } __attribute__((__packed__));

    if (!error)
    {
        for (int ri = desc->first_rx_ring; ri <= desc->last_rx_ring; ri++)
        {
            netmap_ring *ring = NETMAP_RXRING(desc->nifp, ri);
            ring->flags |= NR_FORWARD | NR_TIMESTAMP;
            for (unsigned int i = ring->cur; i != ring->tail; i = nm_ring_next(ring, i))
            {
                auto &slot = ring->slot[i];
                bool used = false;
                // Skip even trying to process packets in the host ring
                if (ri != desc->req.nr_rx_rings
                    && slot.len >= sizeof(header)
                    && !(slot.flags & NS_MOREFRAG))
                {
                    const unsigned char *data = (const unsigned char *) NETMAP_BUF(ring, slot.buf_idx);
                    const header *ph = (const header *) data;
                    /* Checks that this packet is
                     * - big enough
                     * - IPv4, UDP
                     * - unfragmented
                     * - on the right port
                     * It also requires that there are no IP options, since
                     * otherwise the UDP header is at an unknown offset.
                     */
                    if (ph->eth.ether_type == htons(ETHERTYPE_IP)
                        && ph->ip.version == 4
                        && ph->ip.ihl == 5
                        && ph->ip.protocol == IPPROTO_UDP
                        && (ph->ip.frag_off & 0x3f) == 0  /* more fragments bit clear, zero offset */
                        && ph->udp.dest == port_be)
                    {
                        used = true;
                        packet_header packet;
                        const unsigned char *payload = data + sizeof(header);
                        std::size_t payload_size = slot.len - sizeof(header);
                        std::size_t size = decode_packet(packet, payload, payload_size);
                        if (size == payload_size)
                        {
                            get_stream_base().add_packet(packet);
                            if (get_stream_base().is_stopped())
                                log_debug("netmap_udp_reader: end of stream detected");
                        }
                        else if (size != 0)
                        {
                            log_info("discarding packet due to size mismatch (%1% != %2%) flags = %3%",
                                     size, payload_size, slot.flags);
                        }
                    }
                }
                if (!used)
                    slot.flags |= NS_FORWARD;
            }
            ring->cur = ring->head = ring->tail;
        }
    }
    else
        log_warning("error in netmap receive: %1% (%2%)", error.value(), error.message());

    if (get_stream_base().is_stopped())
        stopped();
    else
        enqueue_receive();
}

void netmap_udp_reader::enqueue_receive()
{
    using namespace std::placeholders;
    handle.async_read_some(
        boost::asio::null_buffers(),
        get_stream().get_strand().wrap(std::bind(&netmap_udp_reader::packet_handler, this, _1)));
}

void netmap_udp_reader::stop()
{
    handle.cancel();
}

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_NETMAP
