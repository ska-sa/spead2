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

int udp_ibv_reader::poll_once(stream_base::add_packet_state &state)
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
                    return -2;
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
            stream_base::add_packet_state state(get_stream_base());
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
                int received = poll_once(state);
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
    : udp_ibv_reader(owner, std::vector<boost::asio::ip::udp::endpoint>{endpoint},
                     interface_address, max_size, buffer_size, comp_vector, max_poll)
{
}

static spead2::rdma_cm_id_t make_cm_id(const rdma_event_channel_t &event_channel,
                                       const boost::asio::ip::address &interface_address)
{
    if (!interface_address.is_v4())
        throw std::invalid_argument("interface address is not an IPv4 address");
    rdma_cm_id_t cm_id(event_channel, nullptr, RDMA_PS_UDP);
    cm_id.bind_addr(interface_address);
    return cm_id;
}

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

udp_ibv_reader::udp_ibv_reader(
    stream &owner,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const boost::asio::ip::address &interface_address,
    std::size_t max_size,
    std::size_t buffer_size,
    int comp_vector,
    int max_poll)
    : udp_reader_base(owner),
    cm_id(make_cm_id(event_channel, interface_address)),
    comp_channel_wrapper(owner.get_strand().get_io_service()),
    max_size(max_size),
    n_slots(compute_n_slots(cm_id, buffer_size, max_size + header_length)),
    max_poll(max_poll),
    join_socket(owner.get_strand().get_io_service(), boost::asio::ip::udp::v4()),
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
        throw std::invalid_argument("max_poll must be positive");
    // Re-compute buffer_size as a whole number of slots
    const std::size_t max_raw_size = max_size + header_length;
    buffer_size = n_slots * max_raw_size;

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
    flows = create_flows(qp, endpoints, cm_id->port_num);

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
    for (const auto &endpoint : endpoints)
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
