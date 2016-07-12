/* Copyright 2016 SKA South Africa
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

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#if SPEAD2_USE_IBV
#include <system_error>
#include <stdexcept>
#include <cerrno>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <memory>
#include <algorithm>
#include <boost/asio.hpp>
#include <unistd.h>
#include <spead2/recv_reader.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp_ibv.h>
#include <spead2/common_endian.h>
#include <spead2/common_logging.h>
#include <spead2/common_raw_packet.h>

namespace spead2
{
namespace recv
{

constexpr std::size_t udp_ibv_reader::default_buffer_size;
constexpr int udp_ibv_reader::default_max_poll;
static constexpr int header_length =
    ethernet_frame::min_size + ipv4_packet::min_size + udp_packet::min_size;

ibv_qp_t udp_ibv_reader::create_qp(
    const ibv_pd_t &pd, const ibv_cq_t &send_cq, const ibv_cq_t &recv_cq, std::size_t n_slots)
{
    ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = send_cq.get();
    attr.recv_cq = recv_cq.get();
    attr.qp_type = IBV_QPT_RAW_PACKET;
    attr.cap.max_send_wr = 1;
    attr.cap.max_recv_wr = n_slots;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    return ibv_qp_t(pd, &attr);
}

ibv_flow_t udp_ibv_reader::create_flow(
    const ibv_qp_t &qp, const boost::asio::ip::udp::endpoint &endpoint, int port_num)
{
    struct
    {
        ibv_flow_attr attr;
        ibv_flow_spec_eth eth;
        ibv_flow_spec_ipv4 ip;
        ibv_flow_spec_tcp_udp udp;
    } __attribute__((packed)) flow_rule;
    memset(&flow_rule, 0, sizeof(flow_rule));

    flow_rule.attr.type = IBV_FLOW_ATTR_NORMAL;
    flow_rule.attr.priority = 0;
    flow_rule.attr.size = sizeof(flow_rule);
    flow_rule.attr.num_of_specs = 3;
    flow_rule.attr.port = port_num;

    /* At least the ConnectX-3 cards seem to require an Ethernet match. We
     * thus have to construct the Ethernet multicast address corresponding to
     * the IP multicast address from RFC 7042.
     */
    flow_rule.eth.type = IBV_FLOW_SPEC_ETH;
    flow_rule.eth.size = sizeof(flow_rule.eth);
    mac_address dst_mac = multicast_mac(endpoint.address());
    std::memcpy(&flow_rule.eth.val.dst_mac, &dst_mac, sizeof(dst_mac));
    // Set all 1's mask
    std::memset(&flow_rule.eth.mask.dst_mac, 0xFF, sizeof(flow_rule.eth.mask.dst_mac));

    flow_rule.ip.type = IBV_FLOW_SPEC_IPV4;
    flow_rule.ip.size = sizeof(flow_rule.ip);
    auto bytes = endpoint.address().to_v4().to_bytes(); // big-endian address
    std::memcpy(&flow_rule.ip.val.dst_ip, &bytes, sizeof(bytes));
    std::memset(&flow_rule.ip.mask.dst_ip, 0xFF, sizeof(flow_rule.ip.mask.dst_ip));

    flow_rule.udp.type = IBV_FLOW_SPEC_UDP;
    flow_rule.udp.size = sizeof(flow_rule.udp);
    flow_rule.udp.val.dst_port = htobe16(endpoint.port());
    flow_rule.udp.mask.dst_port = 0xFFFF;

    return ibv_flow_t(qp, &flow_rule.attr);
}

int udp_ibv_reader::poll_once()
{
    int received = recv_cq.poll(n_slots, wc.get());
    for (int i = 0; i < received; i++)
    {
        int index = wc[i].wr_id;
        if (wc[i].status != IBV_WC_SUCCESS)
        {
            log_warning("Work Request failed with code %1%", wc[i].status);
        }
        else
        {
            const void *ptr = reinterpret_cast<void *>(
                reinterpret_cast<std::uintptr_t>(slots[index].sge.addr));
            std::size_t len = wc[i].byte_len;

            // Sanity checks
            try
            {
                ethernet_frame eth(const_cast<void *>(ptr), len);
                if (eth.ethertype() != ipv4_packet::ethertype)
                    log_warning("Frame has wrong ethernet type (VLAN tagging?), discarding");
                else
                {
                    ipv4_packet ipv4 = eth.payload_ipv4();
                    if (ipv4.version() != 4)
                        log_warning("Frame is not IPv4, discarding");
                    else if (ipv4.is_fragment())
                        log_warning("IP datagram is fragmented, discarding");
                    else if (ipv4.protocol() != udp_packet::protocol)
                        log_warning("Packet is not UDP, discarding");
                    else
                    {
                        packet_buffer payload = ipv4.payload_udp().payload();
                        bool stopped = process_one_packet(payload.data(), payload.size(), max_size);
                        if (stopped)
                            return -2;
                    }
                }
            }
            catch (std::length_error &e)
            {
                log_warning(e.what());
            }
        }
        qp.post_recv(&slots[index].wr);
    }
    return received;
}

void udp_ibv_reader::packet_handler(const boost::system::error_code &error)
{
    if (!error)
    {
        if (get_stream_base().is_stopped())
        {
            log_info("UDP reader: discarding packet received after stream stopped");
        }
        else
        {
            if (comp_channel)
            {
                ibv_cq *event_cq;
                void *event_context;
                comp_channel.get_event(&event_cq, &event_context);
                // TODO: defer acks until shutdown
                recv_cq.ack_events(1);
            }
            for (int i = 0; i < max_poll; i++)
            {
                if (comp_channel)
                {
                    if (i == max_poll - 1)
                    {
                        /* We need to call req_notify_cq *before* the last
                         * poll_once, because notifications are edge-triggered.
                         * If we did it the other way around, there is a race
                         * where a new packet can arrive after poll_once but
                         * before req_notify_cq, failing to trigger a
                         * notification.
                         */
                        recv_cq.req_notify(false);
                    }
                }
                else if (stop_poll.load())
                    break;
                int received = poll_once();
                if (received < 0)
                    break;
            }
        }
    }
    else if (error != boost::asio::error::operation_aborted)
        log_warning("Error in UDP receiver: %1%", error.message());

    if (!get_stream_base().is_stopped())
    {
        enqueue_receive();
    }
    else
        stopped();
}

void udp_ibv_reader::enqueue_receive()
{
    using namespace std::placeholders;
    if (comp_channel)
    {
        // Asynchronous mode
        comp_channel_wrapper.async_read_some(
            boost::asio::null_buffers(),
            get_stream().get_strand().wrap(std::bind(&udp_ibv_reader::packet_handler, this, _1)));
    }
    else
    {
        // Polling mode
        get_stream().get_strand().post(std::bind(
                &udp_ibv_reader::packet_handler, this, boost::system::error_code()));
    }
}

udp_ibv_reader::udp_ibv_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    const boost::asio::ip::address &interface_address,
    std::size_t max_size,
    std::size_t buffer_size,
    int comp_vector,
    int max_poll)
    : udp_reader_base(owner),
    max_size(max_size),
    n_slots(std::max(std::size_t(1), buffer_size / (max_size + header_length))),
    max_poll(max_poll),
    join_socket(owner.get_strand().get_io_service(), endpoint.protocol()),
    comp_channel_wrapper(owner.get_strand().get_io_service()),
    stop_poll(false)
{
    if (!endpoint.address().is_v4() || !endpoint.address().is_multicast())
        throw std::invalid_argument("endpoint is not an IPv4 multicast address");
    if (!interface_address.is_v4())
        throw std::invalid_argument("interface address is not an IPv4 address");
    if (max_poll <= 0)
        throw std::invalid_argument("max_poll must be positive");
    // Re-compute buffer_size as a whole number of slots
    const std::size_t max_raw_size = max_size + header_length;
    buffer_size = n_slots * max_raw_size;

    cm_id = rdma_cm_id_t(event_channel, nullptr, RDMA_PS_UDP);
    cm_id.bind_addr(interface_address);

    if (comp_vector >= 0)
    {
        comp_channel = ibv_comp_channel_t(cm_id);
        comp_channel_wrapper = comp_channel.wrap(owner.get_strand().get_io_service());
        recv_cq = ibv_cq_t(cm_id, n_slots, nullptr,
                           comp_channel, comp_vector % cm_id->verbs->num_comp_vectors);
    }
    else
        recv_cq = ibv_cq_t(cm_id, n_slots, nullptr);
    send_cq = ibv_cq_t(cm_id, 1, nullptr);
    pd = ibv_pd_t(cm_id);
    qp = create_qp(pd, send_cq, recv_cq, n_slots);
    qp.modify(IBV_QPS_INIT, cm_id->port_num);
    flow = create_flow(qp, endpoint, cm_id->port_num);

    std::shared_ptr<mmap_allocator> allocator = std::make_shared<mmap_allocator>(0, true);
    buffer = allocator->allocate(buffer_size, nullptr);
    mr = ibv_mr_t(pd, buffer.get(), buffer_size, IBV_ACCESS_LOCAL_WRITE);
    slots.reset(new slot[n_slots]);
    wc.reset(new ibv_wc[n_slots]);
    for (std::size_t i = 0; i < n_slots; i++)
    {
        std::memset(&slots[i], 0, sizeof(slots[i]));
        slots[i].sge.addr = (uintptr_t) &buffer[i * max_raw_size];
        slots[i].sge.length = max_raw_size;
        slots[i].sge.lkey = mr->lkey;
        slots[i].wr.sg_list = &slots[i].sge;
        slots[i].wr.num_sge = 1;
        slots[i].wr.wr_id = i;
        qp.post_recv(&slots[i].wr);
    }

    join_socket.set_option(boost::asio::socket_base::reuse_address(true));
    join_socket.set_option(boost::asio::ip::multicast::join_group(
        endpoint.address().to_v4(), interface_address.to_v4()));

    if (comp_channel)
        recv_cq.req_notify(false);
    enqueue_receive();
    qp.modify(IBV_QPS_RTR);
}

void udp_ibv_reader::stop()
{
    if (comp_channel)
        comp_channel_wrapper.close();
    else
        stop_poll = true;
}

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_IBV
