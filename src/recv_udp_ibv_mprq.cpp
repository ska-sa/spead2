/* Copyright 2019 SKA South Africa
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
#if SPEAD2_USE_IBV_MPRQ
#include <algorithm>
#include <cstdint>
#include <boost/asio.hpp>
#include <spead2/common_raw_packet.h>
#include <spead2/common_ibv.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp_ibv.h>
#include <spead2/recv_udp_ibv_mprq.h>

namespace spead2
{
namespace recv
{

static ibv_qp_t create_qp(const rdma_cm_id_t &cm_id,
                          const ibv_pd_t &pd,
                          const ibv_cq_t &cq,
                          const ibv_exp_rwq_ind_table_t &ind_table)
{
    /* ConnectX-5 only seems to work with Toeplitz hashing. This code seems to
     * work, but I don't really know what I'm doing so it might be horrible.
     */
    uint8_t toeplitz_key[40] = {};
    ibv_exp_rx_hash_conf hash_conf;
    memset(&hash_conf, 0, sizeof(hash_conf));
    hash_conf.rx_hash_function = IBV_EXP_RX_HASH_FUNC_TOEPLITZ;
    hash_conf.rx_hash_key_len = sizeof(toeplitz_key);
    hash_conf.rx_hash_key = toeplitz_key;
    hash_conf.rx_hash_fields_mask = 0;
    hash_conf.rwq_ind_tbl = ind_table.get();

    ibv_exp_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_type = IBV_QPT_RAW_PACKET;
    attr.pd = pd.get();
    attr.rx_hash_conf = &hash_conf;
    attr.port_num = cm_id->port_num;
    attr.comp_mask = IBV_EXP_QP_INIT_ATTR_PD | IBV_EXP_QP_INIT_ATTR_RX_HASH | IBV_EXP_QP_INIT_ATTR_PORT;
    return ibv_qp_t(cm_id, &attr);
}

void udp_ibv_mprq_reader::post_wr(std::size_t offset)
{
    ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t) &buffer[offset];
    sge.length = wqe_size;
    sge.lkey = mr->lkey;
    int status = wq_intf->recv_burst(wq.get(), &sge, 1);
    if (status != 0)
        throw_errno("recv_burst failed");
}

udp_ibv_mprq_reader::poll_result udp_ibv_mprq_reader::poll_once(stream_base::add_packet_state &state)
{
    /* Bound the number of times to receive packets, to avoid live-locking the
     * receive queue if we're getting packets as fast as we can process them.
     */
    const int max_iter = 256;
    for (int iter = 0; iter < max_iter; iter++)
    {
        uint32_t offset;
        uint32_t flags;
        int32_t len = cq_intf->poll_length_flags_mp_rq(recv_cq.get(), &offset, &flags);
        if (len < 0)
        {
            // Error condition.
            ibv_wc wc;
            ibv_poll_cq(recv_cq.get(), 1, &wc);
            log_warning("Work Request failed with code %1%", wc.status);
        }
        else if (len > 0)
        {
            const void *ptr = reinterpret_cast<void *>(buffer.get() + (wqe_start + offset));

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
        if (flags & IBV_EXP_CQ_RX_MULTI_PACKET_LAST_V1)
        {
            post_wr(wqe_start);
            wqe_start += wqe_size;
            if (wqe_start == buffer_size)
                wqe_start = 0;
        }
        if (len == 0 && flags == 0)
            return poll_result::drained;
    }
    return poll_result::partial;
}

udp_ibv_mprq_reader::udp_ibv_mprq_reader(
    stream &owner,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const boost::asio::ip::address &interface_address,
    std::size_t max_size,
    std::size_t buffer_size,
    int comp_vector,
    int max_poll)
    : udp_ibv_reader_base<udp_ibv_mprq_reader>(
        owner, endpoints, interface_address, max_size, comp_vector, max_poll)
{
    ibv_device_attr device_attr = cm_id.query_device();

    // TODO: adjust stride parameters based on device info
    ibv_exp_wq_init_attr wq_attr;
    memset(&wq_attr, 0, sizeof(wq_attr));
    wq_attr.mp_rq.single_stride_log_num_of_bytes = 6;  // 64 bytes per stride
    wq_attr.mp_rq.single_wqe_log_num_of_strides = 14;  // 1MB per WQE
    int log_wqe_size = wq_attr.mp_rq.single_stride_log_num_of_bytes + wq_attr.mp_rq.single_wqe_log_num_of_strides;
    wqe_size = std::size_t(1) << log_wqe_size;
    if (buffer_size < 2 * wqe_size)
        buffer_size = 2 * wqe_size;
    this->buffer_size = buffer_size;

    bool reduced = false;
    std::size_t strides = buffer_size >> wq_attr.mp_rq.single_stride_log_num_of_bytes;
    if (device_attr.max_cqe < strides)
    {
        strides = device_attr.max_cqe;
        reduced = true;
    }
    std::size_t wqe = strides >> wq_attr.mp_rq.single_wqe_log_num_of_strides;
    if (device_attr.max_qp_wr < wqe)
    {
        wqe = device_attr.max_qp_wr;
        reduced = true;
    }
    if (wqe < 2)
        throw std::system_error(std::make_error_code(std::errc::not_supported),
                                "Insufficient resources for a multi-packet receive queue");
    buffer_size = wqe * wqe_size;
    if (reduced)
        log_warning("Reducing buffer to %1% to accommodate device limits", buffer_size);

    if (comp_vector >= 0)
        recv_cq = ibv_cq_t(cm_id, strides, nullptr,
                           comp_channel, comp_vector % cm_id->verbs->num_comp_vectors);
    else
        recv_cq = ibv_cq_t(cm_id, strides, nullptr);
    cq_intf = ibv_exp_cq_family_v1_t(cm_id, recv_cq);

    wq_attr.mp_rq.use_shift = IBV_EXP_MP_RQ_NO_SHIFT;
    wq_attr.max_recv_wr = wqe;
    wq_attr.max_recv_sge = 1;
    wq_attr.pd = pd.get();
    wq_attr.cq = recv_cq.get();
    wq_attr.comp_mask = IBV_EXP_CREATE_WQ_MP_RQ;
    wq = ibv_exp_wq_t(cm_id, &wq_attr);
    wq_intf = ibv_exp_wq_family_t(cm_id, wq);

    rwq_ind_table = create_rwq_ind_table(cm_id, pd, wq);
    qp = create_qp(cm_id, pd, recv_cq, rwq_ind_table);
    wq.modify(IBV_EXP_WQS_RDY);

    std::shared_ptr<mmap_allocator> allocator = std::make_shared<mmap_allocator>(0, true);
    buffer = allocator->allocate(buffer_size, nullptr);
    mr = ibv_mr_t(pd, buffer.get(), buffer_size, IBV_ACCESS_LOCAL_WRITE);
    for (int i = 0; i < wqe; i++)
        post_wr(i * wqe_size);

    flows = create_flows(qp, endpoints, cm_id->port_num);
    enqueue_receive(true);
    join_groups(endpoints, interface_address);
}

udp_ibv_mprq_reader::udp_ibv_mprq_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    const boost::asio::ip::address &interface_address,
    std::size_t max_size,
    std::size_t buffer_size,
    int comp_vector,
    int max_poll)
    : udp_ibv_mprq_reader(owner, std::vector<boost::asio::ip::udp::endpoint>{endpoint},
                          interface_address, max_size, buffer_size, comp_vector, max_poll)
{
}

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_IBV
