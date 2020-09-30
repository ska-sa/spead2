/* Copyright 2016, 2019-2020 SKA South Africa
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

#include <cassert>
#include <spead2/common_raw_packet.h>
#include <spead2/send_udp_ibv.h>

namespace spead2
{
namespace send
{

constexpr std::size_t udp_ibv_stream::default_buffer_size;
constexpr int udp_ibv_stream::default_max_poll;
static constexpr int header_length =
    ethernet_frame::min_size + ipv4_packet::min_size + udp_packet::min_size;

ibv_qp_t udp_ibv_writer::create_qp(
    const ibv_pd_t &pd, const ibv_cq_t &send_cq, const ibv_cq_t &recv_cq, std::size_t n_slots)
{
    ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = send_cq.get();
    attr.recv_cq = recv_cq.get();
    attr.qp_type = IBV_QPT_RAW_PACKET;
    attr.cap.max_send_wr = n_slots;
    attr.cap.max_recv_wr = 0;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 0;
    attr.sq_sig_all = 0;
    return ibv_qp_t(pd, &attr);
}

bool udp_ibv_writer::setup_hw_rate(const ibv_qp_t &qp, const stream_config &config)
{
#if SPEAD2_USE_IBV_HW_RATE_LIMIT
    ibv_device_attr_ex attr;
    if (ibv_query_device_ex(qp->context, nullptr, &attr) != 0)
    {
        log_debug("Not using HW rate limiting because ibv_query_device_ex failed");
        return false;
    }
    if (!ibv_is_qpt_supported(attr.packet_pacing_caps.supported_qpts, IBV_QPT_RAW_PACKET)
        || attr.packet_pacing_caps.qp_rate_limit_max == 0)
    {
        log_debug("Not using HW rate limiting because it is not supported by the device");
        return false;
    }
    // User rate is Bps, for UDP payload. Convert to rate for Ethernet frames, in kbps.
    std::size_t frame_size = config.get_max_packet_size() + 42;
    double overhead = double(frame_size) / config.get_max_packet_size();
    double rate_kbps = config.get_rate() * 8e-3 * overhead;
    if (rate_kbps < attr.packet_pacing_caps.qp_rate_limit_min
        || rate_kbps > attr.packet_pacing_caps.qp_rate_limit_max)
    {
        log_debug("Not using HW rate limiting because the HW does not support the rate");
        return false;
    }

    ibv_qp_rate_limit_attr limit_attr = {};
    limit_attr.rate_limit = rate_kbps;
    limit_attr.typical_pkt_sz = frame_size;
    /* Using config.get_max_burst_size() would cause much longer bursts than
     * necessary if the user did not explicitly turn down the default.
     * Experience with ConnectX-5 shows that it can limit bursts to a single
     * packet with little loss in rate accuracy. If max_burst_sz is set to less
     * than typical_pkt_size it is ignored and a default is used.
     */
    limit_attr.max_burst_sz = frame_size;
    if (ibv_modify_qp_rate_limit(qp.get(), &limit_attr) != 0)
    {
        log_debug("Not using HW rate limiting because ibv_modify_qp_rate_limit failed");
        return false;
    }

    return true;
#else
    log_debug("Not using HW rate limiting because support was not found at compile time");
    return false;
#endif
}

void udp_ibv_writer::reap()
{
    ibv_wc wc;
    int retries = max_poll;
    int heaps = 0;
    std::size_t min_available = std::min(n_slots, available + target_batch);
    // TODO: this could probably be faster with the cq_ex polling API
    // because it could avoid locking.
    while (available < min_available)
    {
        int done = send_cq.poll(1, &wc);
        if (done == 0)
        {
            retries--;
            if (retries < 0)
                break;
            else
                continue;
        }

        boost::system::error_code ec;
        if (wc.status != IBV_WC_SUCCESS)
        {
            log_warning("Work Request failed with code %1%", wc.status);
            // TODO: create some mapping from ibv_wc_status to system error codes
            ec = boost::system::error_code(EIO, boost::asio::error::get_system_category());
        }
        int batch = wc.wr_id;
        for (int i = 0; i < batch; i++)
        {
            const slot *s = &slots[head];
            stream2::queue_item *item = s->item;
            if (ec)
            {
                if (!item->result)
                    item->result = ec;
            }
            else
                item->bytes_sent += s->sge.length - header_length;
            heaps += s->last;
            if (++head == n_slots)
                head = 0;
        }
        available += batch;
    }
    if (heaps > 0)
        heaps_completed(heaps);
}

void udp_ibv_writer::wait_for_space()
{
    if (comp_channel)
    {
        send_cq.req_notify(false);
        auto handler = [this](const boost::system::error_code &, size_t)
        {
            ibv_cq *event_cq;
            void *event_cq_context;
            // This should be non-blocking, since we were woken up, but
            // spurious wakeups have been observed.
            while (comp_channel.get_event(&event_cq, &event_cq_context))
                send_cq.ack_events(1);
            wakeup();
        };
        comp_channel_wrapper.async_read_some(boost::asio::null_buffers(), handler);
    }
    else
        post_wakeup();
}

void udp_ibv_writer::wakeup()
{
    reap();
    if (available < target_batch)
    {
        wait_for_space();
    }
    else
    {
        std::size_t i;
        packet_result result;
        slot *prev = nullptr;
        slot *first = &slots[tail];
        for (i = 0; i < target_batch; i++)
        {
            transmit_packet data;
            result = get_packet(data);
            if (result != packet_result::SUCCESS)
                break;

            slot *s = &slots[tail];
            std::size_t payload_size = data.size;
            ipv4_packet ipv4 = s->frame.payload_ipv4();
            ipv4.total_length(payload_size + udp_packet::min_size + ipv4.header_length());
            udp_packet udp = ipv4.payload_udp();
            udp.length(payload_size + udp_packet::min_size);
            if (get_num_substreams() > 1)
            {
                const std::size_t substream_index = data.item->substream_index;
                const auto &endpoint = endpoints[substream_index];
                s->frame.destination_mac(mac_addresses[substream_index]);
                ipv4.destination_address(endpoint.address().to_v4());
                udp.destination_port(endpoint.port());
            }
            ipv4.update_checksum();
            packet_buffer payload = udp.payload();
            boost::asio::buffer_copy(boost::asio::mutable_buffer(payload), data.pkt.buffers);
            s->sge.length = payload_size + (payload.data() - s->frame.data());
            s->wr.next = nullptr;
            s->wr.send_flags = 0;
            s->item = data.item;
            s->last = data.last;
            if (prev != nullptr)
                prev->wr.next = &s->wr;
            prev = s;

            if (++tail == n_slots)
                tail = 0;
            available--;
        }

        if (i > 0)
        {
            prev->wr.wr_id = i;
            prev->wr.send_flags = IBV_SEND_SIGNALED;
            qp.post_send(&first->wr);
            post_wakeup();
        }
        else if (available < n_slots)
            wait_for_space();
        else if (result == packet_result::SLEEP)
            sleep();
        else
        {
            assert(result == packet_result::EMPTY);
            request_wakeup();
        }
    }
}

static std::size_t calc_n_slots(const stream_config &config, std::size_t buffer_size)
{
    return std::max(std::size_t(1), buffer_size / (config.get_max_packet_size() + header_length));
}

static std::size_t calc_target_batch(const stream_config &config, std::size_t n_slots)
{
    std::size_t packet_size = config.get_max_packet_size() + header_length;
    return std::max(std::size_t(1), std::min(n_slots / 4, 262144 / packet_size));
}

udp_ibv_writer::udp_ibv_writer(
    io_service_ref io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    const boost::asio::ip::address &interface_address,
    std::size_t buffer_size,
    int ttl,
    int comp_vector,
    int max_poll)
    : writer(std::move(io_service), config),
    n_slots(calc_n_slots(config, buffer_size)),
    target_batch(calc_target_batch(config, n_slots)),
    socket(get_io_service(), boost::asio::ip::udp::v4()),
    endpoints(endpoints),
    cm_id(event_channel, nullptr, RDMA_PS_UDP),
    comp_channel_wrapper(get_io_service()),
    available(n_slots),
    max_poll(max_poll)
{
    if (endpoints.empty())
        throw std::invalid_argument("endpoints is empty");
    mac_addresses.reserve(endpoints.size());
    for (const auto &endpoint : endpoints)
    {
        if (!endpoint.address().is_v4() || !endpoint.address().is_multicast())
            throw std::invalid_argument("endpoint is not an IPv4 multicast address");
        mac_addresses.push_back(multicast_mac(endpoint.address()));
    }
    if (!interface_address.is_v4())
        throw std::invalid_argument("interface address is not an IPv4 address");
    if (max_poll <= 0)
        throw std::invalid_argument("max_poll must be positive");
    socket.bind(boost::asio::ip::udp::endpoint(interface_address, 0));
    // Re-compute buffer_size as a whole number of slots
    const std::size_t max_raw_size = config.get_max_packet_size() + header_length;
    buffer_size = n_slots * max_raw_size;

    cm_id.bind_addr(interface_address);
    pd = ibv_pd_t(cm_id);
    if (comp_vector >= 0)
    {
        comp_channel = ibv_comp_channel_t(cm_id);
        comp_channel_wrapper = comp_channel.wrap(get_io_service());
        send_cq = ibv_cq_t(cm_id, n_slots, nullptr,
                           comp_channel, comp_vector % cm_id->verbs->num_comp_vectors);
    }
    else
        send_cq = ibv_cq_t(cm_id, n_slots, nullptr);
    recv_cq = ibv_cq_t(cm_id, 1, nullptr);
    qp = create_qp(pd, send_cq, recv_cq, n_slots);
    qp.modify(IBV_QPS_INIT, cm_id->port_num);
    qp.modify(IBV_QPS_RTR);
    qp.modify(IBV_QPS_RTS);

    if (config.get_allow_hw_rate() && config.get_rate() > 0.0)
    {
        if (setup_hw_rate(qp, config))
            enable_hw_rate();
    }

    std::shared_ptr<mmap_allocator> allocator = std::make_shared<mmap_allocator>(0, true);
    buffer = allocator->allocate(max_raw_size * n_slots, nullptr);
    mr = ibv_mr_t(pd, buffer.get(), buffer_size, IBV_ACCESS_LOCAL_WRITE);
    slots.reset(new slot[n_slots]);
    // We fill in the destination details for the first endpoint. If there are
    // multiple endpoints, they'll get updated for each packet.
    mac_address source_mac = interface_mac(interface_address);
    for (std::size_t i = 0; i < n_slots; i++)
    {
        slots[i].frame = ethernet_frame(buffer.get() + i * max_raw_size, max_raw_size);
        slots[i].sge.addr = (uintptr_t) slots[i].frame.data();
        slots[i].sge.lkey = mr->lkey;
        slots[i].wr.sg_list = &slots[i].sge;
        slots[i].wr.num_sge = 1;
        slots[i].wr.opcode = IBV_WR_SEND;
        slots[i].frame.destination_mac(mac_addresses[0]);
        slots[i].frame.source_mac(source_mac);
        slots[i].frame.ethertype(ipv4_packet::ethertype);
        ipv4_packet ipv4 = slots[i].frame.payload_ipv4();
        ipv4.version_ihl(0x45);  // IPv4, 20 byte header
        // total_length will change later to the actual packet size
        ipv4.total_length(config.get_max_packet_size() + ipv4_packet::min_size + udp_packet::min_size);
        ipv4.flags_frag_off(ipv4_packet::flag_do_not_fragment);
        ipv4.ttl(ttl);
        ipv4.protocol(udp_packet::protocol);
        ipv4.source_address(interface_address.to_v4());
        ipv4.destination_address(endpoints[0].address().to_v4());
        udp_packet udp = ipv4.payload_udp();
        udp.source_port(socket.local_endpoint().port());
        udp.destination_port(endpoints[0].port());
        udp.length(config.get_max_packet_size() + udp_packet::min_size);
        udp.checksum(0);
    }
}


udp_ibv_stream::udp_ibv_stream(
    io_service_ref io_service,
    const boost::asio::ip::udp::endpoint &endpoint,
    const stream_config &config,
    const boost::asio::ip::address &interface_address,
    std::size_t buffer_size,
    int ttl,
    int comp_vector,
    int max_poll)
    : udp_ibv_stream(
        std::move(io_service), std::vector<boost::asio::ip::udp::endpoint>{endpoint},
        config, interface_address, buffer_size, ttl, comp_vector, max_poll)
{
}

udp_ibv_stream::udp_ibv_stream(
    io_service_ref io_service,
    std::initializer_list<boost::asio::ip::udp::endpoint> endpoints,
    const stream_config &config,
    const boost::asio::ip::address &interface_address,
    std::size_t buffer_size,
    int ttl,
    int comp_vector,
    int max_poll)
    : udp_ibv_stream(
        std::move(io_service), std::vector<boost::asio::ip::udp::endpoint>(endpoints),
        config, interface_address, buffer_size, ttl, comp_vector, max_poll)
{
}

udp_ibv_stream::udp_ibv_stream(
    io_service_ref io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    const boost::asio::ip::address &interface_address,
    std::size_t buffer_size,
    int ttl,
    int comp_vector,
    int max_poll)
    : stream2(std::unique_ptr<writer>(new udp_ibv_writer(
        std::move(io_service),
        endpoints,
        config,
        interface_address,
        buffer_size,
        ttl,
        comp_vector,
        max_poll)))
{
}

} // namespace send
} // namespace spead2

#endif // SPEAD2_USE_IBV
