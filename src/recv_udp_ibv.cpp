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
#include "common_features.h"
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
#include "recv_reader.h"
#include "recv_stream.h"
#include "recv_udp_ibv.h"
#include "common_endian.h"
#include "common_logging.h"

namespace spead2
{
namespace recv
{

constexpr std::size_t udp_ibv_reader::default_buffer_size;
constexpr int udp_ibv_reader::default_max_poll;
static constexpr int HEADER_LENGTH = 42; // Eth: 14 IP: 20 UDP: 8

[[noreturn]] static void throw_errno(const char *msg, int err)
{
    /* Many of the ibv_ functions don't explicitly document that errno is
     * set. To protect against the case where it isn't, errno is set to 0
     * first.
     */
    if (err == 0)
    {
        log_warning("%1%: unknown error", msg);
        throw std::system_error(EINVAL, std::system_category());
    }
    else
    {
        std::system_error exception(err, std::system_category());
        log_warning("%1%: %2%", msg, exception.what());
        throw exception;
    }
}

[[noreturn]] static void throw_errno(const char *msg)
{
    throw_errno(msg, errno);
}

std::unique_ptr<rdma_event_channel, detail::rdma_event_channel_deleter>
udp_ibv_reader::create_event_channel()
{
    errno = 0;
    std::unique_ptr<rdma_event_channel, detail::rdma_event_channel_deleter>
        event_channel{rdma_create_event_channel()};
    if (!event_channel)
        throw_errno("rdma_create_event_channel failed");
    return event_channel;
}

std::unique_ptr<rdma_cm_id, detail::rdma_cm_id_deleter>
udp_ibv_reader::create_id(rdma_event_channel *event_channel)
{
    rdma_cm_id *cm_id = nullptr;
    errno = 0;
    int status = rdma_create_id(event_channel, &cm_id, nullptr, RDMA_PS_UDP);
    if (status < 0)
        throw_errno("rdma_create_id failed");
    return std::unique_ptr<rdma_cm_id, detail::rdma_cm_id_deleter>(cm_id);
}

void udp_ibv_reader::bind_address(
    rdma_cm_id *cm_id,
    const boost::asio::ip::address &interface_address)
{
    sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    auto bytes = interface_address.to_v4().to_bytes();
    address.sin_family = AF_INET;
    std::memcpy(&address.sin_addr.s_addr, &bytes, sizeof(address.sin_addr.s_addr));
    errno = 0;
    int status = rdma_bind_addr(cm_id, (sockaddr *) &address);
    if (status < 0)
        throw_errno("rdma_bind_addr failed");
    if (cm_id->verbs == nullptr)
        throw_errno("rdma_bind_addr did not bind to an RDMA device", ENODEV);
}

std::unique_ptr<ibv_comp_channel, detail::ibv_comp_channel_deleter>
udp_ibv_reader::create_comp_channel(ibv_context *context)
{
    errno = 0;
    std::unique_ptr<ibv_comp_channel, detail::ibv_comp_channel_deleter>
        comp_channel(ibv_create_comp_channel(context));
    if (!comp_channel)
        throw_errno("ibv_create_comp_channel failed");
    return comp_channel;
}

boost::asio::posix::stream_descriptor udp_ibv_reader::wrap_comp_channel(
    boost::asio::io_service &io_service, ibv_comp_channel *comp_channel)
{
    int fd = dup(comp_channel->fd);
    if (fd < 0)
        throw_errno("dup failed");
    boost::asio::posix::stream_descriptor descriptor(io_service, fd);
    descriptor.native_non_blocking(true);
    return descriptor;
}

std::unique_ptr<ibv_cq, detail::ibv_cq_deleter>
udp_ibv_reader::create_cq(
    ibv_context *context, int cqe, ibv_comp_channel *comp_channel, int comp_vector)
{
    errno = 0;
    std::unique_ptr<ibv_cq, detail::ibv_cq_deleter>
        cq(ibv_create_cq(context, cqe, nullptr, comp_channel, comp_vector));
    if (!cq)
        throw_errno("ibv_create_cq failed");
    return cq;
}

std::unique_ptr<ibv_pd, detail::ibv_pd_deleter>
udp_ibv_reader::create_pd(ibv_context *context)
{
    errno = 0;
    std::unique_ptr<ibv_pd, detail::ibv_pd_deleter>
        pd(ibv_alloc_pd(context));
    if (!pd)
        throw_errno("ibv_alloc_pd failed");
    return pd;
}

std::unique_ptr<ibv_qp, detail::ibv_qp_deleter>
udp_ibv_reader::create_qp(ibv_pd *pd, ibv_cq *send_cq, ibv_cq *recv_cq, std::size_t n_slots)
{
    ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = send_cq;
    attr.recv_cq = recv_cq;
    attr.qp_type = IBV_QPT_RAW_PACKET;
    attr.cap.max_send_wr = 1;
    attr.cap.max_recv_wr = n_slots;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    errno = 0;
    std::unique_ptr<ibv_qp, detail::ibv_qp_deleter> qp(ibv_create_qp(pd, &attr));
    if (!qp)
        throw_errno("ibv_create_qp failed");
    return qp;
}

std::unique_ptr<ibv_mr, detail::ibv_mr_deleter>
udp_ibv_reader::create_mr(ibv_pd *pd, void *addr, std::size_t length)
{
    errno = 0;
    std::unique_ptr<ibv_mr, detail::ibv_mr_deleter>
        mr(ibv_reg_mr(pd, addr, length, IBV_ACCESS_LOCAL_WRITE));
    if (!mr)
        throw_errno("ibv_reg_mr failed");
    return mr;
}

void udp_ibv_reader::init_qp(ibv_qp *qp, int port_num)
{
    ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = port_num;
    int status = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PORT);
    if (status != 0)
        throw_errno("ibv_modify_qp failed", status);
}

std::unique_ptr<ibv_flow, detail::ibv_flow_deleter>
udp_ibv_reader::create_flow(ibv_qp *qp, const boost::asio::ip::udp::endpoint &endpoint,
                            int port_num)
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
    auto bytes = endpoint.address().to_v4().to_bytes(); // big-endian address
    flow_rule.eth.type = IBV_FLOW_SPEC_ETH;
    flow_rule.eth.size = sizeof(flow_rule.eth);
    std::memcpy(&flow_rule.eth.val.dst_mac[2], &bytes, sizeof(bytes));
    flow_rule.eth.val.dst_mac[0] = 0x01;
    flow_rule.eth.val.dst_mac[1] = 0x00;
    flow_rule.eth.val.dst_mac[2] = 0x5e;
    flow_rule.eth.val.dst_mac[3] &= 0x7f;
    // Set all 1's mask
    std::memset(&flow_rule.eth.mask.dst_mac, 0xFF, sizeof(flow_rule.eth.mask.dst_mac));

    flow_rule.ip.type = IBV_FLOW_SPEC_IPV4;
    flow_rule.ip.size = sizeof(flow_rule.ip);
    std::memcpy(&flow_rule.ip.val.dst_ip, &bytes, sizeof(bytes));
    std::memset(&flow_rule.ip.mask.dst_ip, 0xFF, sizeof(flow_rule.ip.mask.dst_ip));

    flow_rule.udp.type = IBV_FLOW_SPEC_UDP;
    flow_rule.udp.size = sizeof(flow_rule.udp);
    flow_rule.udp.val.dst_port = htobe16(endpoint.port());
    flow_rule.udp.mask.dst_port = 0xFFFF;

    errno = 0;
    std::unique_ptr<ibv_flow, detail::ibv_flow_deleter>
        flow(ibv_create_flow(qp, &flow_rule.attr));
    if (!flow)
        throw_errno("ibv_create_flow failed");
    return flow;
}

void udp_ibv_reader::rtr_qp(ibv_qp *qp)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    int status = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
    if (status != 0)
        throw_errno("ibv_modify_qp to IBV_QPS_RTR failed", status);
}

void udp_ibv_reader::req_notify_cq(ibv_cq *cq)
{
    int status = ibv_req_notify_cq(cq, 0);
    if (status != 0)
        throw_errno("ibv_req_notify_cq failed", status);
}

void udp_ibv_reader::post_slot(std::size_t index)
{
    ibv_recv_wr *bad_wr;
    int status = ibv_post_recv(qp.get(), &slots[index].wr, &bad_wr);
    if (status != 0)
        throw_errno("ibv_post_recv failed", status);
}

int udp_ibv_reader::poll_once()
{
    int received = ibv_poll_cq(recv_cq.get(), n_slots, wc.get());
    if (received < 0)
    {
        log_warning("ibv_poll_cq failed");
        return -1;
    }
    for (int i = 0; i < received; i++)
    {
        int index = wc[i].wr_id;
        if (wc[i].status != IBV_WC_SUCCESS)
        {
            log_warning("Work Request failed with code %1%", wc[i].status);
        }
        else
        {
            const std::uint8_t *ptr = reinterpret_cast<std::uint8_t *>(
                reinterpret_cast<std::uintptr_t>(slots[index].sge.addr));
            std::size_t len = wc[i].byte_len;

            constexpr int HEADER_LENGTH = 42; // Eth: 14 IP: 20 UDP: 8
            constexpr std::uint8_t ethertype_ipv4[2] = {0x08, 0x00};
            // Sanity checks
            if (len <= HEADER_LENGTH)
                log_warning("Frame is too short to contain UDP payload, discarding");
            else if (std::memcmp(ethertype_ipv4, ptr + 12, 2))
                log_warning("Frame has wrong ethernet type (VLAN tagging?), discarding");
            else if (ptr[14] != 0x45)
                log_warning("Frame is not IPv4 or has extra options, discarding");
            else if ((ptr[20] & 0x3f) || (ptr[21] != 0)) // flags and fragment offset
                log_warning("IP message is fragmented, discarding");
            else
            {
                len -= HEADER_LENGTH;
                ptr += HEADER_LENGTH;
                bool stopped = process_one_packet(ptr, len, max_size);
                if (stopped)
                    return -2;
            }
        }
        post_slot(index);
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
                int status = ibv_get_cq_event(comp_channel.get(), &event_cq, &event_context);
                if (status < 0)
                    log_warning("ibv_get_cq_event failed");
                else
                {
                    // TODO: defer acks until shutdown
                    ibv_ack_cq_events(event_cq, 1);
                }
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
                        req_notify_cq(recv_cq.get());
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
    n_slots(std::max(std::size_t(1), buffer_size / (max_size + HEADER_LENGTH))),
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
    const int max_raw_size = max_size + HEADER_LENGTH;
    buffer_size = n_slots * max_raw_size;

    event_channel = create_event_channel();
    cm_id = create_id(event_channel.get());
    bind_address(cm_id.get(), interface_address);
    ibv_context *context = cm_id->verbs;
    assert(context);

    if (comp_vector >= 0)
    {
        comp_channel = create_comp_channel(context);
        comp_channel_wrapper = wrap_comp_channel(owner.get_strand().get_io_service(), comp_channel.get());
        recv_cq = create_cq(context, n_slots, comp_channel.get(), comp_vector % context->num_comp_vectors);
    }
    else
        recv_cq = create_cq(context, n_slots, nullptr, 0);
    send_cq = create_cq(context, 1, nullptr, 0);
    pd = create_pd(context);
    qp = create_qp(pd.get(), send_cq.get(), recv_cq.get(), n_slots);
    init_qp(qp.get(), cm_id->port_num);
    flow = create_flow(qp.get(), endpoint, cm_id->port_num);

    buffer.reset(new std::uint8_t[buffer_size]);
    mr = create_mr(pd.get(), buffer.get(), buffer_size);
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
        post_slot(i);
    }

    join_socket.set_option(boost::asio::socket_base::reuse_address(true));
    join_socket.set_option(boost::asio::ip::multicast::join_group(
        endpoint.address().to_v4(), interface_address.to_v4()));

    if (comp_channel)
        req_notify_cq(recv_cq.get());
    enqueue_receive();
    rtr_qp(qp.get());
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
