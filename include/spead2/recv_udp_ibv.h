/* Copyright 2016, 2019 SKA South Africa
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
#include <utility>
#include <vector>
#include <boost/asio.hpp>
#include <boost/noncopyable.hpp>
#include <spead2/common_ibv.h>
#include <spead2/common_logging.h>
#include <spead2/recv_reader.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp_base.h>

namespace spead2
{
namespace recv
{

namespace detail
{

/* Parts of udp_ibv_reader_base that don't need to be templated */
class udp_ibv_reader_core : public udp_reader_base
{
private:
    /**
     * Socket that is used only to join the multicast group. It is not
     * bound to a port.
     */
    boost::asio::ip::udp::socket join_socket;

protected:
    enum class poll_result
    {
        stopped,       ///< stream was stopped
        partial,       ///< read zero or more CQEs, but stopped to prevent livelock
        drained,       ///< CQ fully drained
    };

    // Data structures required by ibverbs
    rdma_event_channel_t event_channel;
    rdma_cm_id_t cm_id;
    ibv_pd_t pd;
    ibv_comp_channel_t comp_channel;
    boost::asio::posix::stream_descriptor comp_channel_wrapper;
    std::vector<ibv_flow_t> flows;
    ibv_cq_t recv_cq;

    ///< Maximum supported packet size
    const std::size_t max_size;
    ///< Number of times to poll before waiting
    const int max_poll;
    /// Signals poll-mode to stop
    std::atomic<bool> stop_poll;

    void join_groups(const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
                     const boost::asio::ip::address &interface_address);

public:
    /// Receive buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 16 * 1024 * 1024;
    /// Number of times to poll in a row, if none is explicitly passed to the constructor
    static constexpr int default_max_poll = 10;

    udp_ibv_reader_core(
        stream &owner,
        const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
        const boost::asio::ip::address &interface_address,
        std::size_t max_size,
        int comp_vector,
        int max_poll);

    virtual void stop() override;
};

/**
 * Common code between spead2::recv::udp_ibv_reader and
 * spead2::recv::udp_ibv_mprq_reader. It uses the curiously recursive template
 * pattern to avoid virtual functions.
 */
template<typename Derived>
class udp_ibv_reader_base : public udp_ibv_reader_core
{
protected:
    /**
     * Retrieve packets from the completion queue and process them.
     *
     * This is called from the io_service either when the completion channel
     * is notified (non-polling mode) or by a post to the io_service (polling
     * mode).
     *
     * If @a consume_event is true, an event should be removed and consumed
     * from the completion channel.
     */
    void packet_handler(const boost::system::error_code &error,
                        bool consume_event);

    /**
     * Request a callback when there is data (or as soon as possible, in
     * polling mode or when @a need_poll is true).
     */
    void enqueue_receive(bool needs_poll);

    using udp_ibv_reader_core::udp_ibv_reader_core;
};

template<typename Derived>
void udp_ibv_reader_base<Derived>::packet_handler(const boost::system::error_code &error,
                                                  bool consume_event)
{
    stream_base::add_packet_state state(get_stream_base());

    bool need_poll = true;
    if (!error)
    {
        if (consume_event)
        {
            ibv_cq *event_cq;
            void *event_context;
            comp_channel.get_event(&event_cq, &event_context);
            // TODO: defer acks until shutdown
            recv_cq.ack_events(1);
        }
        if (state.is_stopped())
        {
            log_info("UDP reader: discarding packet received after stream stopped");
        }
        else
        {
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
                        need_poll = false;
                    }
                }
                else if (stop_poll.load())
                    break;
                poll_result result = static_cast<Derived *>(this)->poll_once(state);
                if (result == poll_result::stopped)
                    break;
                else if (result == poll_result::partial)
                {
                    /* If we armed req_notify_cq but then didn't drain the CQ, and
                     * we get no more packets, then we won't get woken up again, so
                     * we need to poll again next time we go around the event loop.
                     */
                    need_poll = true;
                }
            }
        }
    }
    else if (error != boost::asio::error::operation_aborted)
        log_warning("Error in UDP receiver: %1%", error.message());

    if (!state.is_stopped())
    {
        enqueue_receive(need_poll);
    }
    else
        stopped();
}

template<typename Derived>
void udp_ibv_reader_base<Derived>::enqueue_receive(bool need_poll)
{
    using namespace std::placeholders;
    if (comp_channel && !need_poll)
    {
        // Asynchronous mode
        comp_channel_wrapper.async_read_some(
            boost::asio::null_buffers(),
            std::bind(&udp_ibv_reader_base<Derived>::packet_handler, this, _1, true));
    }
    else
    {
        // Polling mode
        get_io_service().post(
            std::bind(&udp_ibv_reader_base<Derived>::packet_handler, this,
                      boost::system::error_code(), false));
    }
}

} // namespace detail

/**
 * Synchronous or asynchronous stream reader that reads UDP packets using
 * the Infiniband verbs API. It currently only supports multicast IPv4, with
 * no fragmentation, IP header options, or VLAN tags.
 */
class udp_ibv_reader : public detail::udp_ibv_reader_base<udp_ibv_reader>
{
private:
    friend class detail::udp_ibv_reader_base<udp_ibv_reader>;

    struct slot : boost::noncopyable
    {
        ibv_recv_wr wr;
        ibv_sge sge;
    };

    // All the data structures required by ibverbs
    ibv_cq_t send_cq;
    ibv_qp_t qp;
    ibv_mr_t mr;

    ///< Number of packets that can be queued
    const std::size_t n_slots;

    /// Data buffer for all the packets
    memory_allocator::pointer buffer;

    /// array of @ref n_slots slots for work requests
    std::unique_ptr<slot[]> slots;
    /// array of @ref n_slots work completions
    std::unique_ptr<ibv_wc[]> wc;

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
    udp_ibv_reader(
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
    udp_ibv_reader(
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

#include <spead2/recv_udp_ibv_mprq.h>

namespace spead2
{
namespace recv
{

template<>
struct reader_factory<udp_ibv_reader>
{
    template<typename... Args>
    static std::unique_ptr<reader> make_reader(Args&&... args)
    {
        /* Note: using perfect forwarding twice on the same args is normally a
         * bad idea if any of them are rvalue references. But the constructors
         * we're forwarding to don't have any.
         */
#if SPEAD2_USE_IBV_MPRQ
        try
        {
            std::unique_ptr<reader> reader(new udp_ibv_mprq_reader(
                std::forward<Args>(args)...));
            log_info("Using multi-packet receive queue for verbs acceleration");
            return reader;
        }
        catch (std::system_error &e)
        {
            if (e.code() != std::errc::not_supported)
                throw;
            log_debug("Multi-packet receive queues not supported (%1%), falling back", e.what());
            return std::unique_ptr<reader>(new udp_ibv_reader(
                std::forward<Args>(args)...));
        }
#else
        return std::unique_ptr<reader>(new udp_ibv_reader(
            std::forward<Args>(args)...));
#endif
    }
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_IBV

#endif // SPEAD2_RECV_UDP_IBV_H
