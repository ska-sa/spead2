/* Copyright 2019-2020, 2023 National Research Foundation (SARAO)
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

#ifndef SPEAD2_RECV_UDP_IBV_MPRQ_H
#define SPEAD2_RECV_UDP_IBV_MPRQ_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#if SPEAD2_USE_MLX5DV
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <boost/asio.hpp>
#include <spead2/common_ibv.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp_base.h>
#include <spead2/recv_udp_ibv.h>

namespace spead2::recv
{

/**
 * Synchronous or asynchronous stream reader that reads UDP packets using
 * the Infiniband verbs API with multi-packet receive queues. It currently only
 * supports multicast IPv4, with no fragmentation, IP header options, or VLAN
 * tags.
 */
class udp_ibv_mprq_reader : public detail::udp_ibv_reader_base<udp_ibv_mprq_reader>
{
private:
    friend class detail::udp_ibv_reader_base<udp_ibv_mprq_reader>;

    // All the data structures required by ibverbs
    ibv_cq_ex_t recv_cq;
    ibv_wq_mprq_t wq;
    ibv_rwq_ind_table_t rwq_ind_table;
    ibv_qp_t qp;
    ibv_mr_t mr;
    std::vector<ibv_flow_t> flows;

    /// Data buffer for all the packets
    memory_allocator::pointer buffer;

    /// Bytes of buffer for each WQ entry
    std::size_t wqe_size;
    /// Buffer offset of the current WQE
    std::size_t wqe_start = 0;
    /// Total buffer size
    std::size_t buffer_size;

    /// Post one work request to the receive work queue
    void post_wr(std::size_t offset);

    /// Do one pass over the completion queue.
    poll_result poll_once(stream_base::add_packet_state &state);

public:
    /**
     * Constructor.
     *
     * @param owner        Owning stream
     * @param config       Configuration
     *
     * @throws std::invalid_argument If no endpoints are set.
     * @throws std::invalid_argument If no interface address is set.
     */
    udp_ibv_mprq_reader(stream &owner, const udp_ibv_config &config);
};

} // namespace spead2::recv

#endif // SPEAD2_USE_IBV

#endif // SPEAD2_RECV_UDP_IBV_H
