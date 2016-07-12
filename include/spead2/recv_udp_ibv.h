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

#ifndef SPEAD2_RECV_UDP_IBV_H
#define SPEAD2_RECV_UDP_IBV_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#if SPEAD2_USE_IBV
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <boost/asio.hpp>
#include <boost/noncopyable.hpp>
#include <spead2/common_ibv.h>
#include <spead2/recv_reader.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp_base.h>

namespace spead2
{
namespace recv
{

/**
 * Synchronous or asynchronous stream reader that reads UDP packets using
 * the Infiniband verbs API. It currently only supports multicast IPv4, with
 * no fragmentation, IP header options, or VLAN tags.
 */
class udp_ibv_reader : public udp_reader_base
{
private:
    struct slot : boost::noncopyable
    {
        ibv_recv_wr wr;
        ibv_sge sge;
    };

    ///< Maximum supported packet size
    const std::size_t max_size;
    ///< Number of packets that can be queued
    const std::size_t n_slots;
    ///< Number of times to poll before waiting
    const int max_poll;

    /**
     * Socket that is used only to join the multicast group. It is not
     * bound to a port.
     */
    boost::asio::ip::udp::socket join_socket;
    /// Data buffer for all the packets
    memory_allocator::pointer buffer;

    // All the data structures required by ibverbs
    rdma_event_channel_t event_channel;
    rdma_cm_id_t cm_id;
    ibv_pd_t pd;
    ibv_comp_channel_t comp_channel;
    boost::asio::posix::stream_descriptor comp_channel_wrapper;
    ibv_cq_t send_cq;
    ibv_cq_t recv_cq;
    ibv_qp_t qp;
    ibv_flow_t flow;
    ibv_mr_t mr;

    /// array of @ref n_slots slots for work requests
    std::unique_ptr<slot[]> slots;
    /// array of @ref n_slots work completions
    std::unique_ptr<ibv_wc[]> wc;
    /// Signals poll-mode to stop
    std::atomic<bool> stop_poll;

    // Utility functions to create the data structures
    static ibv_qp_t
    create_qp(const ibv_pd_t &pd, const ibv_cq_t &send_cq, const ibv_cq_t &recv_cq,
              std::size_t n_slots);

    static ibv_flow_t
    create_flow(const ibv_qp_t &qp, const boost::asio::ip::udp::endpoint &endpoint,
                int port_num);

    static void req_notify_cq(ibv_cq *cq);

    /**
     * Do one pass over the completion queue.
     *
     * @retval -1 if there was an ibverbs failure
     * @retval -2 if the stream received a stop packet
     * @retval n otherwise, where n is the number of packets received
     */
    int poll_once();

    /**
     * Retrieve packets from the completion queue and process them.
     *
     * This is called from the io_service either when the completion channel
     * is notified (non-polling mode) or by a post to the strand (polling
     * mode).
     */
    void packet_handler(const boost::system::error_code &error);

    /**
     * Request a callback when there is data (or as soon as possible, in
     * polling mode).
     */
    void enqueue_receive();

public:
    /// Receive buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 16 * 1024 * 1024;
    /// Number of times to poll in a row, if none is explicitly passed to the constructor
    static constexpr int default_max_poll = 10;

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
    udp_ibv_reader(
        stream &owner,
        const boost::asio::ip::udp::endpoint &endpoint,
        const boost::asio::ip::address &interface_address,
        std::size_t max_size = default_max_size,
        std::size_t buffer_size = default_buffer_size,
        int comp_vector = 0,
        int max_poll = default_max_poll);

    virtual void stop() override;
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_IBV

#endif // SPEAD2_RECV_UDP_IBV_H
