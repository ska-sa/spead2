/* Copyright 2016-2020 National Research Foundation (SARAO)
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

#ifndef SPEAD2_COMMON_IBV_H
#define SPEAD2_COMMON_IBV_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#include <spead2/common_logging.h>
#include <spead2/common_loader_rdmacm.h>
#include <spead2/common_loader_ibv.h>
#include <spead2/common_loader_mlx5dv.h>
#include <memory>
#include <vector>
#include <cstdint>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <boost/asio.hpp>
#include <system_error>

#if SPEAD2_USE_IBV

#if SPEAD2_USE_MLX5DV
# include <infiniband/mlx5dv.h>
#endif

namespace spead2
{

namespace detail
{

// Deleters for unique_ptr wrappers of the various ibverbs structures

struct rdma_cm_id_deleter
{
    void operator()(rdma_cm_id *cm_id)
    {
        rdma_destroy_id(cm_id);
    }
};

struct rdma_event_channel_deleter
{
    void operator()(rdma_event_channel *event_channel)
    {
        rdma_destroy_event_channel(event_channel);
    }
};

struct ibv_context_deleter
{
    void operator()(ibv_context *ctx)
    {
        ibv_close_device(ctx);
    }
};

struct ibv_qp_deleter
{
    void operator()(ibv_qp *qp)
    {
        ibv_destroy_qp(qp);
    }
};

struct ibv_cq_deleter
{
    void operator()(ibv_cq *cq)
    {
        ibv_destroy_cq(cq);
    }
};

struct ibv_mr_deleter
{
    void operator()(ibv_mr *mr)
    {
        ibv_dereg_mr(mr);
    }
};

struct ibv_pd_deleter
{
    void operator()(ibv_pd *pd)
    {
        ibv_dealloc_pd(pd);
    }
};

struct ibv_comp_channel_deleter
{
    void operator()(ibv_comp_channel *comp_channel)
    {
        ibv_destroy_comp_channel(comp_channel);
    }
};

struct ibv_flow_deleter
{
    void operator()(ibv_flow *flow);
};

struct ibv_wq_deleter
{
    void operator()(ibv_wq *wq)
    {
        ibv_destroy_wq(wq);
    }
};

struct ibv_rwq_ind_table_deleter
{
    void operator()(ibv_rwq_ind_table *table)
    {
        ibv_destroy_rwq_ind_table(table);
    }
};

} // namespace detail

class rdma_event_channel_t : public std::unique_ptr<rdma_event_channel, detail::rdma_event_channel_deleter>
{
public:
    rdma_event_channel_t();
    // Allow explicitly creating an uninitialized unique_ptr
    rdma_event_channel_t(std::nullptr_t) {}
};

class rdma_cm_id_t : public std::unique_ptr<rdma_cm_id, detail::rdma_cm_id_deleter>
{
public:
    rdma_cm_id_t() = default;
    rdma_cm_id_t(const rdma_event_channel_t &cm_id, void *context, rdma_port_space ps);

    void bind_addr(const boost::asio::ip::address &addr);
    ibv_device_attr query_device() const;  // for backwards compatibility
    ibv_device_attr_ex query_device_ex(const ibv_query_device_ex_input *input = nullptr) const;
#if SPEAD2_USE_MLX5DV
    bool mlx5dv_is_supported() const;
    mlx5dv_context mlx5dv_query_device() const;
#endif
};

/* This class is not intended to be used for anything. However, the mlx5 driver
 * will only enable multicast loopback if there at least 2 device contexts, and
 * multiple instances of rdma_cm_id_t bound to the same device end up with the
 * same device context, so constructing one is a way to force multicast
 * loopback to function.
 */
class ibv_context_t : public std::unique_ptr<ibv_context, detail::ibv_context_deleter>
{
public:
    ibv_context_t() = default;
    explicit ibv_context_t(struct ibv_device *device);
    explicit ibv_context_t(const boost::asio::ip::address &addr);
};

class ibv_comp_channel_t : public std::unique_ptr<ibv_comp_channel, detail::ibv_comp_channel_deleter>
{
public:
    ibv_comp_channel_t() = default;
    explicit ibv_comp_channel_t(const rdma_cm_id_t &cm_id);

    /// Create a file descriptor that is ready to read when the completion channel has events
    boost::asio::posix::stream_descriptor wrap(boost::asio::io_service &io_service) const;
    /// Get an event, if one is available
    bool get_event(ibv_cq **cq, void **context);
};

class ibv_cq_t : public std::unique_ptr<ibv_cq, detail::ibv_cq_deleter>
{
public:
    ibv_cq_t() = default;
    ibv_cq_t(const rdma_cm_id_t &cm_id, int cqe, void *context);
    ibv_cq_t(const rdma_cm_id_t &cm_id, int cqe, void *context,
             const ibv_comp_channel_t &comp_channel, int comp_vector);

    void req_notify(bool solicited_only);
    int poll(int num_entries, ibv_wc *wc);
    void ack_events(unsigned int nevents);
};

class ibv_cq_ex_t : public ibv_cq_t
{
public:
    ibv_cq_ex_t() = default;
    ibv_cq_ex_t(const rdma_cm_id_t &cm_id, ibv_cq_init_attr_ex *cq_attr);

    ibv_cq_ex *get() const
    {
        return (ibv_cq_ex *) ibv_cq_t::get();
    }

    ibv_cq_ex *operator->() const
    {
        return get();
    }

    /* RAII wrapper around ibv_start_poll / ibv_next_poll / ibv_end_poll.
     * It's implemented inline for efficiency. Note that it does NOT take
     * ownership of the CQ: the caller must keep the CQ alive.
     */
    class poller
    {
    private:
        ibv_cq_ex *cq;
        bool active = false;

    public:
        poller(const ibv_cq_ex_t &cq)
            : cq(cq.get())
        {
            assert(this->cq);
        }

        bool next()
        {
            if (!active)
            {
                ibv_poll_cq_attr attr = {};
                int result = ibv_start_poll(cq, &attr);
                if (result == 0)
                {
                    active = true;
                    return true;
                }
                else if (result == ENOENT)
                    return false;
                else
                    throw_errno("ibv_start_poll failed", result);
            }
            else
            {
                int result = ibv_next_poll(cq);
                if (result == 0)
                    return true;
                else if (result == ENOENT)
                    return false;
                else
                    throw_errno("ibv_next_poll failed", result);
            }
        }

        void stop()
        {
            if (active)
            {
                ibv_end_poll(cq);
                active = false;
            }
        }

        ~poller()
        {
            stop();
        }
    };
};

class ibv_pd_t : public std::unique_ptr<ibv_pd, detail::ibv_pd_deleter>
{
public:
    ibv_pd_t() = default;
    explicit ibv_pd_t(const rdma_cm_id_t &cm_id);
};

class ibv_qp_t : public std::unique_ptr<ibv_qp, detail::ibv_qp_deleter>
{
public:
    ibv_qp_t() = default;
    ibv_qp_t(const ibv_pd_t &pd, ibv_qp_init_attr *init_attr);
    ibv_qp_t(const rdma_cm_id_t &cm_id, ibv_qp_init_attr_ex *init_attr);

    void modify(ibv_qp_attr *attr, int attr_mask);
    void modify(ibv_qp_state qp_state);
    void modify(ibv_qp_state qp_state, int port_num);

    // bad_wr is ignored, because we throw an exception on failure
    void post_recv(ibv_recv_wr *wr);
    void post_send(ibv_send_wr *wr);
};

class ibv_mr_t : public std::unique_ptr<ibv_mr, detail::ibv_mr_deleter>
{
public:
    ibv_mr_t() = default;
    /* If allow_relaxed_ordering is true, it will try to set
     * IBV_ACCESS_RELAXED_ORDERING if supported, but fall back gracefully if
     * not.
     */
    ibv_mr_t(const ibv_pd_t &pd, void *addr, std::size_t length, int access,
             bool allow_relaxed_ordering = true);
};

class ibv_flow_t : public std::unique_ptr<ibv_flow, detail::ibv_flow_deleter>
{
public:
    ibv_flow_t() = default;
    ibv_flow_t(const ibv_qp_t &qp, ibv_flow_attr *flow);
};

/**
 * Create a single flow rule.
 *
 * @pre The endpoint contains an IPv4 address.
 */
ibv_flow_t create_flow(
    const ibv_qp_t &qp, const boost::asio::ip::udp::endpoint &endpoint,
    int port_num);

/**
 * Create flow rules to subscribe to a given set of endpoints.
 *
 * If the address in an endpoint is unspecified, it will not be filtered on.
 * Multicast addresses are supported; unicast addresses must have corresponding
 * interfaces (which are used to retrieve the corresponding MAC addresses).
 *
 * @pre The @a endpoints are IPv4 addresses.
 */
std::vector<ibv_flow_t> create_flows(
    const ibv_qp_t &qp, const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    int port_num);

class ibv_wq_t : public std::unique_ptr<ibv_wq, detail::ibv_wq_deleter>
{
public:
    ibv_wq_t() = default;
    ibv_wq_t(const rdma_cm_id_t &cm_id, ibv_wq_init_attr *attr);

    void modify(ibv_wq_state state);
};

#if SPEAD2_USE_MLX5DV
/**
 * Support for multi-packet receive queues (MPRQs) on mlx5. At this time (Sep
 * 2020), libmlx5 doesn't support them through the standard verbs functions. We
 * thus have to directly construct the low-level data structures for the
 * hardware, and decode information stored in the completion entries.
 *
 * This is somewhat fragile, because we're not taking over management of the
 * completion queue, and the completion queue code assumes it is working with
 * a normal work queue. Examining the code shows that it is probably safe,
 * provided that
 * - Only a single thread is accessing these data structures at a time (this
 *   class does no locking.
 * - There is no inline data in the CQE (I don't know how to ensure that).
 *
 * The @a mlx5_attr must specify @c MLX5DV_WQ_INIT_ATTR_MASK_STRIDING_RQ.
 */
class ibv_wq_mprq_t : public ibv_wq_t
{
private:
    mlx5dv_rwq rwq;
    /// Total number of entries posted
    std::uint32_t head = 0;
    /// Total number of entries fully consumed
    std::uint32_t tail = 0;
    /// Number of strides consumed from the tail entry
    std::uint32_t tail_strides = 0;
    /// Size of each stride in bytes
    std::uint32_t stride_size = 0;
    /// Number of strides per work queue entry
    std::uint32_t n_strides = 0;
    /// Offset of actual packet data within its first stride
    std::int32_t data_offset = 0;
public:
    static constexpr int FLAG_LAST = 1;    ///< This is the last CQE for the WQE
    static constexpr int FLAG_FILLER = 2;  ///< This is a filler CQE rather than a packet

    ibv_wq_mprq_t() = default;
    ibv_wq_mprq_t(const rdma_cm_id_t &cm_id, ibv_wq_init_attr *attr, mlx5dv_wq_init_attr *mlx5_attr);

    void post_recv(ibv_sge *sge);

    /**
     * Get completion information. Call this function from inside
     * ibv_start_poll/ibv_end_poll, exactly once for each completion.
     */
    void read_wc(const ibv_cq_ex_t &cq, std::uint32_t &byte_len, std::uint32_t &offset, int &flags) noexcept;
};
#endif // SPEAD2_USE_MLX5DV

class ibv_rwq_ind_table_t : public std::unique_ptr<ibv_rwq_ind_table, detail::ibv_rwq_ind_table_deleter>
{
public:
    ibv_rwq_ind_table_t() = default;
    ibv_rwq_ind_table_t(const rdma_cm_id_t &cm_id, ibv_rwq_ind_table_init_attr *attr);
};

/// Construct a table with a single entry
ibv_rwq_ind_table_t create_rwq_ind_table(const rdma_cm_id_t &cm_id, const ibv_wq_t &wq);

namespace detail
{

/**
 * Base for @ref spead2::recv::udp_ibv_config and @ref spead2::send::udp_ibv_config.
 * It uses the curiously recursive template pattern so that the setters can return
 * references to the derived class.
 */
template<typename Derived>
class udp_ibv_config_base
{
private:
    std::vector<boost::asio::ip::udp::endpoint> endpoints;
    boost::asio::ip::address interface_address;
    std::size_t buffer_size = Derived::default_buffer_size;
    int comp_vector = 0;
    int max_poll = Derived::default_max_poll;

public:
    /// Get the configured endpoints
    const std::vector<boost::asio::ip::udp::endpoint> &get_endpoints() const { return endpoints; }
    /**
     * Set the endpoints (replacing any previous).
     *
     * @throws std::invalid_argument if any element of @a endpoints is invalid.
     */
    Derived &set_endpoints(const std::vector<boost::asio::ip::udp::endpoint> &endpoints);
    /**
     * Append a single endpoint.
     *
     * @throws std::invalid_argument if @a endpoint is invalid.
     */
    Derived &add_endpoint(const boost::asio::ip::udp::endpoint &endpoint);

    /// Get the currently set interface address
    const boost::asio::ip::address get_interface_address() const { return interface_address; }
    /**
     * Set the interface address.
     *
     * @throws std::invalid_argument if @a interface_address is not an IPv4 address.
     */
    Derived &set_interface_address(const boost::asio::ip::address &interface_address);

    /// Get the currently configured buffer size.
    std::size_t get_buffer_size() const { return buffer_size; }
    /**
     * Set the buffer size.
     *
     * The value 0 is special and resets it to the default. The actual buffer size
     * used may be slightly different to round it to a whole number of
     * packet-sized slots.
     */
    Derived &set_buffer_size(std::size_t buffer_size);

    /// Get the completion channel vector (see @ref set_comp_vector)
    int get_comp_vector() const { return comp_vector; }
    /**
     * Set the completion channel vector (interrupt) for asynchronous operation.
     * Use a negative value to poll continuously. Polling should not be used if
     * there are other users of the thread pool. If a non-negative value is
     * provided, it is taken modulo the number of available completion vectors.
     * This allows a number of streams to be assigned sequential completion
     * vectors and have them load-balanced, without concern for the number
     * available.
     */
    Derived &set_comp_vector(int comp_vector);

    /// Get maximum number of times to poll in a row (see @ref set_max_poll)
    int get_max_poll() const { return max_poll; }
    /**
     * Set maximum number of times to poll in a row.
     *
     * If interrupts are enabled (default), it is the maximum number of times
     * to poll before waiting for an interrupt; if they are disabled by @ref
     * set_comp_vector, it is the number of times to poll before letting other
     * code run on the thread.
     *
     * @throws std::invalid_argument if @a max_poll is zero.
     */
    Derived &set_max_poll(int max_poll);
};

template<typename Derived>
Derived &udp_ibv_config_base<Derived>::set_endpoints(
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints)
{
    for (const auto &endpoint : endpoints)
        Derived::validate_endpoint(endpoint);
    this->endpoints = endpoints;
    return *static_cast<Derived *>(this);
}

template<typename Derived>
Derived &udp_ibv_config_base<Derived>::add_endpoint(
    const boost::asio::ip::udp::endpoint &endpoint)
{
    Derived::validate_endpoint(endpoint);
    endpoints.push_back(endpoint);
    return *static_cast<Derived *>(this);
}

template<typename Derived>
Derived &udp_ibv_config_base<Derived>::set_interface_address(
    const boost::asio::ip::address &interface_address)
{
    if (!interface_address.is_v4())
        throw std::invalid_argument("interface address is not an IPv4 address");
    this->interface_address = interface_address;
    return *static_cast<Derived *>(this);
}

template<typename Derived>
Derived &udp_ibv_config_base<Derived>::set_buffer_size(std::size_t buffer_size)
{
    if (buffer_size == 0)
        this->buffer_size = Derived::default_buffer_size;
    else
        this->buffer_size = buffer_size;
    return *static_cast<Derived *>(this);
}

template<typename Derived>
Derived &udp_ibv_config_base<Derived>::set_comp_vector(int comp_vector)
{
    this->comp_vector = comp_vector;
    return *static_cast<Derived *>(this);
}

template<typename Derived>
Derived &udp_ibv_config_base<Derived>::set_max_poll(int max_poll)
{
    if (max_poll < 1)
        throw std::invalid_argument("max_poll must be at least 1");
    this->max_poll = max_poll;
    return *static_cast<Derived *>(this);
}

} // namespace detail

} // namespace spead2

#endif // SPEAD2_USE_IBV

#endif // SPEAD2_COMMON_IBV_H
