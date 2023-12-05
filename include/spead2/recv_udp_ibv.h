/* Copyright 2016, 2019-2020, 2023 National Research Foundation (SARAO)
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
#include <spead2/common_defines.h>
#include <spead2/common_ibv.h>
#include <spead2/common_logging.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_udp_base.h>

namespace spead2
{

// Prevent the compiler instantiating the template in all translation units
// (we'll explicitly instantiate it in recv_udp_ibv.cpp).
namespace recv { class udp_ibv_config; }
extern template class detail::udp_ibv_config_base<recv::udp_ibv_config>;

namespace recv
{

/**
 * Configuration for @ref udp_ibv_reader.
 */
class udp_ibv_config : public spead2::detail::udp_ibv_config_base<udp_ibv_config>
{
public:
    /// Receive buffer size, if none is explicitly set
    static constexpr std::size_t default_buffer_size = 16 * 1024 * 1024;
    /// Maximum packet size to accept, if none is explicitly set
    static constexpr std::size_t default_max_size = udp_reader_base::default_max_size;
    /// Number of times to poll in a row, if none is explicitly set
    static constexpr int default_max_poll = 10;

private:
    std::size_t max_size = default_max_size;

    friend class spead2::detail::udp_ibv_config_base<udp_ibv_config>;
    static void validate_endpoint(const boost::asio::ip::udp::endpoint &endpoint);

public:
    /// Get maximum packet size to accept
    std::size_t get_max_size() const { return max_size; }

    /// Set maximum packet size to accept
    udp_ibv_config &set_max_size(std::size_t max_size);
};

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
    /// Multicast groups to subscribe to
    std::vector<boost::asio::ip::address> groups;
    /// Interface address for multicast subscription
    boost::asio::ip::address interface_address;

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

    ///< Maximum supported packet size
    const std::size_t max_size;
    ///< Number of times to poll before waiting
    const int max_poll;

    void join_groups();

public:
    udp_ibv_reader_core(
        stream &owner,
        const udp_ibv_config &config);

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
    void packet_handler(
        handler_context ctx,
        stream_base::add_packet_state &state,
        const boost::system::error_code &error,
        bool consume_event);

    /**
     * Request a callback when there is data (or as soon as possible, in
     * polling mode or when @a need_poll is true).
     */
    void enqueue_receive(handler_context ctx, bool needs_poll);

    using udp_ibv_reader_core::udp_ibv_reader_core;
};

template<typename Derived>
void udp_ibv_reader_base<Derived>::packet_handler(
    handler_context ctx,
    stream_base::add_packet_state &state,
    const boost::system::error_code &error,
    bool consume_event)
{
    bool need_poll = true;
    if (!error)
    {
        if (consume_event)
        {
            ibv_cq *event_cq;
            void *event_context;
            while (comp_channel.get_event(&event_cq, &event_context))
            {
                // TODO: defer acks until shutdown
                // TODO: make both cases use ibv_cq_ex_t so that recv_cq can
                // move back into base.
                static_cast<Derived *>(this)->recv_cq.ack_events(1);
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
                    static_cast<Derived *>(this)->recv_cq.req_notify(false);
                    need_poll = false;
                }
            }
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
    else if (error != boost::asio::error::operation_aborted)
        log_warning("Error in UDP receiver: %1%", error.message());

    if (!state.is_stopped())
    {
        enqueue_receive(std::move(ctx), need_poll);
    }
}

template<typename Derived>
void udp_ibv_reader_base<Derived>::enqueue_receive(handler_context ctx, bool need_poll)
{
    using namespace std::placeholders;
    if (comp_channel && !need_poll)
    {
        // Asynchronous mode
        comp_channel_wrapper.async_wait(
            comp_channel_wrapper.wait_read,
            bind_handler(
                std::move(ctx),
                std::bind(&udp_ibv_reader_base<Derived>::packet_handler, this, _1, _2, _3, true)));
    }
    else
    {
        // Polling mode
        boost::asio::post(
            get_io_service(),
            bind_handler(
                std::move(ctx),
                std::bind(&udp_ibv_reader_base<Derived>::packet_handler, this, _1, _2,
                          boost::system::error_code(), false)
            )
        );
    }
}

} // namespace detail

/**
 * Synchronous or asynchronous stream reader that reads UDP packets using
 * the Infiniband verbs API. It currently only supports IPv4, with
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
    ibv_cq_t recv_cq;
    ibv_qp_t qp;
    ibv_mr_t mr;
    /* Note: don't try to move this to the base class, even though it is
     * shared with udp_ibv_reader_mprq. It needs to be destroyed before the
     * QP, otherwise destroying the QP fails with EBUSY.
     */
    std::vector<ibv_flow_t> flows;

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
     * @param config       Configuration
     *
     * @throws std::invalid_argument If no endpoints are set.
     * @throws std::invalid_argument If no interface address is set.
     */
    udp_ibv_reader(stream &owner, const udp_ibv_config &config);

    virtual void start() override;
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
#if SPEAD2_USE_MLX5DV
        try
        {
            auto reader = std::make_unique<udp_ibv_mprq_reader>(std::forward<Args>(args)...);
            log_info("Using multi-packet receive queue for verbs acceleration");
            return reader;
        }
        catch (std::system_error &e)
        {
            if (e.code() != std::errc::not_supported                // ENOTSUP
                && e.code() != std::errc::function_not_supported)   // ENOSYS
                throw;
            log_debug("Multi-packet receive queues not supported (%1%), falling back", e.what());
            return std::make_unique<udp_ibv_reader>(std::forward<Args>(args)...);
        }
#else
        return std::make_unique<udp_ibv_reader>(std::forward<Args>(args)...);
#endif
    }
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_USE_IBV

#endif // SPEAD2_RECV_UDP_IBV_H
