/* Copyright 2019-2020 National Research Foundation (SARAO)
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
#if SPEAD2_USE_MLX5DV
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <utility>
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
                          const ibv_rwq_ind_table_t &ind_table)
{
    /* ConnectX-5 only seems to work with Toeplitz hashing. This code seems to
     * work, but I don't really know what I'm doing so it might be horrible.
     */
    uint8_t toeplitz_key[40] = {};

    ibv_qp_init_attr_ex attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_type = IBV_QPT_RAW_PACKET;
    attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_IND_TABLE | IBV_QP_INIT_ATTR_RX_HASH;
    attr.pd = pd.get();
    attr.rwq_ind_tbl = ind_table.get();
    attr.rx_hash_conf.rx_hash_function = IBV_RX_HASH_FUNC_TOEPLITZ;
    attr.rx_hash_conf.rx_hash_key_len = sizeof(toeplitz_key);
    attr.rx_hash_conf.rx_hash_key = toeplitz_key;
    attr.rx_hash_conf.rx_hash_fields_mask = 0;
    return ibv_qp_t(cm_id, &attr);
}

void udp_ibv_mprq_reader::post_wr(std::size_t offset)
{
    ibv_sge sge;
    sge.addr = (uintptr_t) &buffer[offset];
    sge.length = wqe_size;
    sge.lkey = mr->lkey;
    wq.post_recv(&sge);
}

udp_ibv_mprq_reader::poll_result udp_ibv_mprq_reader::poll_once(stream_base::add_packet_state &state)
{
    /* Bound the number of times to receive packets, to avoid live-locking the
     * receive queue if we're getting packets as fast as we can process them.
     */
    const int max_iter = 256;
    ibv_cq_ex_t::poller poller(recv_cq);

    for (int iter = 0; iter < max_iter; iter++)
    {
        if (!poller.next())
            return poll_result::drained;
        if (recv_cq->status != IBV_WC_SUCCESS)
        {
            log_warning("Work Request failed with code %1%", recv_cq->status);
            continue;
        }

        std::uint32_t len, offset;
        int flags;
        wq.read_wc(recv_cq, len, offset, flags);

        if (!(flags & ibv_wq_mprq_t::FLAG_FILLER))
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
        if (flags & ibv_wq_mprq_t::FLAG_LAST)
        {
            /* Temporarily stop the poller so that we release the CQ entries
             * we've consumed so far. Not doing this risks an overflow if
             * packets are very small.
             */
            poller.stop();
            post_wr(wqe_start);
            wqe_start += wqe_size;
            if (wqe_start == buffer_size)
                wqe_start = 0;
        }
    }
    return poll_result::partial;
}

static int clamp(int x, int low, int high)
{
    return std::min(std::max(x, low), high);
}

udp_ibv_mprq_reader::udp_ibv_mprq_reader(
    stream &owner,
    const udp_ibv_config &config)
    : udp_ibv_reader_base<udp_ibv_mprq_reader>(owner, config)
{
    if (!cm_id.mlx5dv_is_supported())
        throw std::system_error(std::make_error_code(std::errc::not_supported),
                                "device does not support mlx5dv API");
    mlx5dv_context mlx5dv_attr = cm_id.mlx5dv_query_device();
    if (!(mlx5dv_attr.comp_mask & MLX5DV_CONTEXT_MASK_STRIDING_RQ)
        || !(mlx5dv_attr.flags & MLX5DV_CONTEXT_FLAGS_MPW_ALLOWED)
        || !ibv_is_qpt_supported(mlx5dv_attr.striding_rq_caps.supported_qpts, IBV_QPT_RAW_PACKET))
        throw std::system_error(std::make_error_code(std::errc::not_supported),
                                "device does not support multi-packet receive queues");

    mlx5dv_wq_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.comp_mask = MLX5DV_WQ_INIT_ATTR_MASK_STRIDING_RQ;
    attr.striding_rq_attrs.single_stride_log_num_of_bytes =
        clamp(6,
              mlx5dv_attr.striding_rq_caps.min_single_stride_log_num_of_bytes,
              mlx5dv_attr.striding_rq_caps.max_single_stride_log_num_of_bytes);   // 64 bytes per stride
    attr.striding_rq_attrs.single_wqe_log_num_of_strides =
        clamp(20 - attr.striding_rq_attrs.single_stride_log_num_of_bytes,
              mlx5dv_attr.striding_rq_caps.min_single_wqe_log_num_of_strides,
              mlx5dv_attr.striding_rq_caps.max_single_wqe_log_num_of_strides);    // 1MB per WQE
    int log_wqe_size = attr.striding_rq_attrs.single_stride_log_num_of_bytes
        + attr.striding_rq_attrs.single_wqe_log_num_of_strides;
    wqe_size = std::size_t(1) << log_wqe_size;
    std::size_t buffer_size = config.get_buffer_size();
    if (buffer_size < 2 * wqe_size)
        buffer_size = 2 * wqe_size;

    bool reduced = false;
    std::size_t strides = buffer_size >> attr.striding_rq_attrs.single_stride_log_num_of_bytes;
    ibv_device_attr device_attr = cm_id.query_device();
    if (std::size_t(device_attr.max_cqe) < strides)
    {
        strides = device_attr.max_cqe;
        reduced = true;
    }
    std::size_t wqe = strides >> attr.striding_rq_attrs.single_wqe_log_num_of_strides;
    if (std::size_t(device_attr.max_qp_wr) < wqe)
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
    this->buffer_size = buffer_size;

    ibv_cq_init_attr_ex cq_attr;
    memset(&cq_attr, 0, sizeof(cq_attr));
    cq_attr.cqe = strides;
    cq_attr.wc_flags = IBV_WC_EX_WITH_BYTE_LEN;
    if (config.get_comp_vector() >= 0)
    {
        cq_attr.channel = comp_channel.get();
        cq_attr.comp_vector = config.get_comp_vector() % cm_id->verbs->num_comp_vectors;
    }
    cq_attr.flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED;
    cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
    recv_cq = ibv_cq_ex_t(cm_id, &cq_attr);

    ibv_wq_init_attr wq_attr;
    memset(&wq_attr, 0, sizeof(wq_attr));
    wq_attr.wq_type = IBV_WQT_RQ;
    wq_attr.max_wr = wqe;
    wq_attr.max_sge = 1;
    wq_attr.pd = pd.get();
    wq_attr.cq = ibv_cq_ex_to_cq(recv_cq.get());
    // TODO: investigate IBV_WQ_FLAGS_DELAY_DROP to reduce dropped
    // packets and IBV_WQ_FLAGS_CVLAN_STRIPPING to remove VLAN tags.
    wq = ibv_wq_mprq_t(cm_id, &wq_attr, &attr);

    rwq_ind_table = create_rwq_ind_table(cm_id, wq);
    qp = create_qp(cm_id, pd, rwq_ind_table);
    wq.modify(IBV_WQS_RDY);

    std::shared_ptr<mmap_allocator> allocator = std::make_shared<mmap_allocator>(0, true);
    buffer = allocator->allocate(buffer_size, nullptr);
    mr = ibv_mr_t(pd, buffer.get(), buffer_size, IBV_ACCESS_LOCAL_WRITE);
    for (std::size_t i = 0; i < wqe; i++)
        post_wr(i * wqe_size);

    flows = create_flows(qp, config.get_endpoints(), cm_id->port_num);
    enqueue_receive(true);
    join_groups(config.get_endpoints(), config.get_interface_address());
}

udp_ibv_mprq_reader::udp_ibv_mprq_reader(
    stream &owner,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const boost::asio::ip::address &interface_address,
    std::size_t max_size,
    std::size_t buffer_size,
    int comp_vector,
    int max_poll)
    : udp_ibv_mprq_reader(
        owner,
        udp_ibv_config()
            .set_endpoints(endpoints)
            .set_interface_address(interface_address)
            .set_max_size(max_size)
            .set_buffer_size(buffer_size)
            .set_comp_vector(comp_vector)
            .set_max_poll(max_poll))
{
}

udp_ibv_mprq_reader::udp_ibv_mprq_reader(
    stream &owner,
    const boost::asio::ip::udp::endpoint &endpoint,
    const boost::asio::ip::address &interface_address,
    std::size_t max_size,
    std::size_t buffer_size,
    int comp_vector,
    int max_poll)
    : udp_ibv_mprq_reader(
        owner,
        udp_ibv_config()
            .add_endpoint(endpoint)
            .set_interface_address(interface_address)
            .set_max_size(max_size)
            .set_buffer_size(buffer_size)
            .set_comp_vector(comp_vector)
            .set_max_poll(max_poll))
{
}

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_IBV
