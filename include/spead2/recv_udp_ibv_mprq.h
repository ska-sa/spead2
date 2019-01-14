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

#ifndef SPEAD2_RECV_UDP_IBV_MPRQ_H
#define SPEAD2_RECV_UDP_IBV_MPRQ_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#if SPEAD2_USE_IBV_MPRQ
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <boost/asio.hpp>
#include <spead2/common_ibv.h>
#include <spead2/recv_reader.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp_base.h>
#include <spead2/recv_udp_ibv.h>

namespace spead2
{
namespace recv
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
    ibv_exp_res_domain_t res_domain;
    ibv_exp_wq_t wq;
    ibv_exp_rwq_ind_table_t rwq_ind_table;
    ibv_exp_cq_family_v1_t cq_intf;
    ibv_exp_wq_family_t wq_intf;
    ibv_qp_t qp;
    ibv_mr_t mr;

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
     * @param endpoint     Multicast group and port
     * @param max_size     Maximum packet size that will be accepted
     * @param buffer_size  Requested memory allocation for work requests. Note
     *                     that this is used to determine the number of packets
     *                     to buffer; if the packets are smaller than @a max_size,
     *                     then fewer bytes will be buffered.
     * @param interface_address  Address of the interface which should join the group and listen for data
     * @param comp_vector  Completion channel vector (interrupt) for asynchronous operation, or
     *                     a negative value to poll continuously. Polling
     *                     should not be used if there are other users of the
     *                     thread pool. If a non-negative value is provided, it
     *                     is taken modulo the number of available completion
     *                     vectors. This allows a number of readers to be
     *                     assigned sequential completion vectors and have them
     *                     load-balanced, without concern for the number
     *                     available.
     * @param max_poll     Maximum number of times to poll in a row, without
     *                     waiting for an interrupt (if @a comp_vector is
     *                     non-negative) or letting other code run on the
     *                     thread (if @a comp_vector is negative).
     *
     * @throws std::invalid_argument If @a endpoint is not an IPv4 multicast address
     * @throws std::invalid_argument If @a interface_address is not an IPv4 address
     */
    udp_ibv_mprq_reader(
        stream &owner,
        const boost::asio::ip::udp::endpoint &endpoint,
        const boost::asio::ip::address &interface_address,
        std::size_t max_size = default_max_size,
        std::size_t buffer_size = default_buffer_size,
        int comp_vector = 0,
        int max_poll = default_max_poll);

    /**
     * Constructor with multiple endpoints.
     *
     * @param owner        Owning stream
     * @param endpoints    Multicast groups and ports
     * @param max_size     Maximum packet size that will be accepted
     * @param buffer_size  Requested memory allocation for work requests. Note
     *                     that this is used to determine the number of packets
     *                     to buffer; if the packets are smaller than @a max_size,
     *                     then fewer bytes will be buffered.
     * @param interface_address  Address of the interface which should join the group and listen for data
     * @param comp_vector  Completion channel vector (interrupt) for asynchronous operation, or
     *                     a negative value to poll continuously. Polling
     *                     should not be used if there are other users of the
     *                     thread pool. If a non-negative value is provided, it
     *                     is taken modulo the number of available completion
     *                     vectors. This allows a number of readers to be
     *                     assigned sequential completion vectors and have them
     *                     load-balanced, without concern for the number
     *                     available.
     * @param max_poll     Maximum number of times to poll in a row, without
     *                     waiting for an interrupt (if @a comp_vector is
     *                     non-negative) or letting other code run on the
     *                     thread (if @a comp_vector is negative).
     *
     * @throws std::invalid_argument If any element of @a endpoints is not an IPv4 multicast address
     * @throws std::invalid_argument If @a interface_address is not an IPv4 address
     */
    udp_ibv_mprq_reader(
        stream &owner,
        const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
        const boost::asio::ip::address &interface_address,
        std::size_t max_size = default_max_size,
        std::size_t buffer_size = default_buffer_size,
        int comp_vector = 0,
        int max_poll = default_max_poll);
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_IBV

#endif // SPEAD2_RECV_UDP_IBV_H
