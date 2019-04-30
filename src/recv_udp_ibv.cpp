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
#include <sstream>
#include <cerrno>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>
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

namespace detail
{

constexpr std::size_t udp_ibv_reader_core::default_buffer_size;
constexpr int udp_ibv_reader_core::default_max_poll;
static constexpr int header_length =
    ethernet_frame::min_size + ipv4_packet::min_size + udp_packet::min_size;

static spead2::rdma_cm_id_t make_cm_id(const rdma_event_channel_t &event_channel,
                                       const boost::asio::ip::address &interface_address)
{
    if (!interface_address.is_v4())
        throw std::invalid_argument("interface address is not an IPv4 address");
    rdma_cm_id_t cm_id(event_channel, nullptr, RDMA_PS_UDP);
    cm_id.bind_addr(interface_address);
    return cm_id;
}

udp_ibv_reader_core::udp_ibv_reader_core(
    stream &owner,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const boost::asio::ip::address &interface_address,
    std::size_t max_size,
    int comp_vector,
    int max_poll)
    : udp_reader_base(owner),
    join_socket(owner.get_io_service(), boost::asio::ip::udp::v4()),
    comp_channel_wrapper(owner.get_io_service()),
    max_size(max_size),
    max_poll(max_poll),
    stop_poll(false)
{
    for (const auto &endpoint : endpoints)
        if (!endpoint.address().is_v4() || !endpoint.address().is_multicast())
        {
            std::ostringstream msg;
            msg << "endpoint " << endpoint << " is not an IPv4 multicast address";
            throw std::invalid_argument(msg.str());
        }
    if (max_poll <= 0)
        throw std::invalid_argument("max_poll must be non-negative");

    cm_id = make_cm_id(event_channel, interface_address);
    pd = ibv_pd_t(cm_id);
    if (comp_vector >= 0)
    {
        comp_channel = ibv_comp_channel_t(cm_id);
        comp_channel_wrapper = comp_channel.wrap(get_io_service());
    }
}

void udp_ibv_reader_core::join_groups(
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const boost::asio::ip::address &interface_address)
{
    join_socket.set_option(boost::asio::socket_base::reuse_address(true));
    for (const auto &endpoint : endpoints)
        join_socket.set_option(boost::asio::ip::multicast::join_group(
            endpoint.address().to_v4(), interface_address.to_v4()));
}

void udp_ibv_reader_core::stop()
{
    if (comp_channel)
        comp_channel_wrapper.close();
    else
        stop_poll = true;
}

} // namespace detail

static std::size_t compute_n_slots(const rdma_cm_id_t &cm_id, std::size_t buffer_size,
                                   std::size_t max_raw_size)
{
    bool reduced = false;
    ibv_device_attr attr = cm_id.query_device();
    if (attr.max_mr_size < max_raw_size)
        throw std::invalid_argument("Packet size is larger than biggest MR supported by device");
    if (attr.max_mr_size < buffer_size)
    {
        buffer_size = attr.max_mr_size;
        reduced = true;
    }

    std::size_t n_slots = std::max(std::size_t(1), buffer_size / max_raw_size);
    std::size_t hw_slots = std::min(attr.max_qp_wr, attr.max_cqe);
    if (hw_slots == 0)
        throw std::invalid_argument("This device does not have a usable verbs implementation");
    if (hw_slots < n_slots)
    {
        n_slots = hw_slots;
        reduced = true;
    }
    if (reduced)
        log_warning("Reducing buffer to %1% to accommodate device limits", n_slots * max_raw_size);
    return n_slots;
}

static ibv_qp_t create_qp(
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

udp_ibv_reader::poll_result udp_ibv_reader::poll_once(stream_base::add_packet_state &state)
{
    /* Number of work requests to queue at a time. This is a balance between
     * not calling post_recv too often (it takes a lock) and not starving the
     * receive queue.
     */
    const std::size_t post_batch = std::min(std::max(n_slots / 4, std::size_t(1)), std::size_t(64));
    int received = recv_cq.poll(n_slots, wc.get());
    ibv_recv_wr *head = nullptr, *tail = nullptr;
    std::size_t cur_batch = 0;
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
                packet_buffer payload = udp_from_ethernet(const_cast<void *>(ptr), len);
                bool stopped = process_one_packet(state,
                                                  payload.data(), payload.size(), max_size);
                if (stopped)
                    return poll_result::stopped;
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
        if (tail == nullptr)
            head = tail = &slots[index].wr;
        else
            tail = tail->next = &slots[index].wr;
        cur_batch++;
        if (cur_batch == post_batch)
        {
            tail->next = nullptr;
            qp.post_recv(head);
            cur_batch = 0;
            head = tail = nullptr;
        }
    }
    if (head != nullptr)
    {
        tail->next = nullptr;
        qp.post_recv(head);
    }
    return poll_result::drained;
}

udp_ibv_reader::udp_ibv_reader(
    stream &owner,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const boost::asio::ip::address &interface_address,
    std::size_t max_size,
    std::size_t buffer_size,
    int comp_vector,
    int max_poll)
    : udp_ibv_reader_base<udp_ibv_reader>(
        owner, endpoints, interface_address, max_size, comp_vector, max_poll),
    n_slots(compute_n_slots(cm_id, buffer_size, max_size + detail::header_length))
{
    // Re-compute buffer_size as a whole number of slots
    const std::size_t max_raw_size = max_size + detail::header_length;
    buffer_size = n_slots * max_raw_size;

    if (comp_vector >= 0)
        recv_cq = ibv_cq_t(cm_id, n_slots, nullptr,
                           comp_channel, comp_vector % cm_id->verbs->num_comp_vectors);
    else
        recv_cq = ibv_cq_t(cm_id, n_slots, nullptr);
    send_cq = ibv_cq_t(cm_id, 1, nullptr);
    qp = create_qp(pd, send_cq, recv_cq, n_slots);
    qp.modify(IBV_QPS_INIT, cm_id->port_num);
    flows = create_flows(qp, endpoints, cm_id->port_num);

    std::shared_ptr<mmap_allocator> allocator = std::make_shared<mmap_allocator>(0, true);
    buffer = allocator->allocate(buffer_size, nullptr);
    mr = ibv_mr_t(pd, buffer.get(), buffer_size, IBV_ACCESS_LOCAL_WRITE);
    slots.reset(new slot[n_slots]);
    wc.reset(new ibv_wc[n_slots]);
    for (std::size_t i = 0; i < n_slots; i++)
    {
        slots[i].sge.addr = (uintptr_t) &buffer[i * max_raw_size];
        slots[i].sge.length = max_raw_size;
        slots[i].sge.lkey = mr->lkey;
        std::memset(&slots[i].wr, 0, sizeof(slots[i].wr));
        slots[i].wr.sg_list = &slots[i].sge;
        slots[i].wr.num_sge = 1;
        slots[i].wr.wr_id = i;
        qp.post_recv(&slots[i].wr);
    }

    enqueue_receive(true);
    qp.modify(IBV_QPS_RTR);
    join_groups(endpoints, interface_address);
}

udp_ibv_reader::udp_ibv_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    const boost::asio::ip::address &interface_address,
    std::size_t max_size,
    std::size_t buffer_size,
    int comp_vector,
    int max_poll)
    : udp_ibv_reader(owner, std::vector<boost::asio::ip::udp::endpoint>{endpoint},
                     interface_address, max_size, buffer_size, comp_vector, max_poll)
{
}

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_IBV
