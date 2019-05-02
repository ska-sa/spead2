/* Copyright 2016-2019 SKA South Africa
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
#include <spead2/common_ibv_loader.h>
#include <memory>
#include <vector>
#include <cstdint>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <boost/asio.hpp>
#include <system_error>

#if SPEAD2_USE_IBV

#if SPEAD2_USE_IBV_EXP
# include <infiniband/verbs_exp.h>
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
    void operator()(ibv_flow *flow)
    {
        ibv_destroy_flow(flow);
    }
};

#if SPEAD2_USE_IBV_MPRQ

struct ibv_exp_wq_deleter
{
    void operator()(ibv_exp_wq *wq)
    {
        ibv_exp_destroy_wq(wq);
    }
};

struct ibv_exp_rwq_ind_table_deleter
{
    void operator()(ibv_exp_rwq_ind_table *table)
    {
        ibv_exp_destroy_rwq_ind_table(table);
    }
};

class ibv_intf_deleter
{
private:
    struct ibv_context *context;

public:
    explicit ibv_intf_deleter(struct ibv_context *context = nullptr) noexcept;
    void operator()(void *intf);
};

class ibv_exp_res_domain_deleter
{
private:
    struct ibv_context *context;

public:
    explicit ibv_exp_res_domain_deleter(struct ibv_context *context = nullptr) noexcept;
    void operator()(ibv_exp_res_domain *res_domain);
};

#endif

} // namespace detail

class rdma_event_channel_t : public std::unique_ptr<rdma_event_channel, detail::rdma_event_channel_deleter>
{
public:
    rdma_event_channel_t();
};

class rdma_cm_id_t : public std::unique_ptr<rdma_cm_id, detail::rdma_cm_id_deleter>
{
public:
    rdma_cm_id_t() = default;
    rdma_cm_id_t(const rdma_event_channel_t &cm_id, void *context, rdma_port_space ps);

    void bind_addr(const boost::asio::ip::address &addr);
    ibv_device_attr query_device() const;
#if SPEAD2_USE_IBV_EXP
    ibv_exp_device_attr exp_query_device() const;
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
    void get_event(ibv_cq **cq, void **context);
};

class ibv_cq_t : public std::unique_ptr<ibv_cq, detail::ibv_cq_deleter>
{
public:
    ibv_cq_t() = default;
    ibv_cq_t(const rdma_cm_id_t &cm_id, int cqe, void *context);
    ibv_cq_t(const rdma_cm_id_t &cm_id, int cqe, void *context,
             const ibv_comp_channel_t &comp_channel, int comp_vector);
#if SPEAD2_USE_IBV_EXP
    ibv_cq_t(const rdma_cm_id_t &cm_id, int cqe, void *context,
             ibv_exp_cq_init_attr *attr);
    ibv_cq_t(const rdma_cm_id_t &cm_id, int cqe, void *context,
             const ibv_comp_channel_t &comp_channel, int comp_vector,
             ibv_exp_cq_init_attr *attr);
#endif

    void req_notify(bool solicited_only);
    int poll(int num_entries, ibv_wc *wc);
    void ack_events(unsigned int nevents);

#if SPEAD2_USE_IBV_EXP
    int poll(int num_entries, ibv_exp_wc *wc);
#endif
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
#if SPEAD2_USE_IBV_MPRQ
    ibv_qp_t(const rdma_cm_id_t &cm_id, ibv_exp_qp_init_attr *init_attr);
#endif

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
    ibv_mr_t(const ibv_pd_t &pd, void *addr, std::size_t length, int access);
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
 * The @a mask specifies a subset of bits in the IPv4 address that must match
 * (in host byte order, so for example 0xFFFFFF00 specifies a /24 subnet).
 * Note that not all IB drivers support a mask other than the default (all
 * 1's).
 *
 * @pre The endpoint contains an IPv4 multicast address.
 */
ibv_flow_t create_flow(
    const ibv_qp_t &qp, const boost::asio::ip::udp::endpoint &endpoint,
    int port_num, std::uint32_t mask = 0xFFFFFFFF);

/**
 * Create flow rules to subscribe to a given set of multicast endpoints.
 *
 * Where supported by the driver, it will coalesce multiple endpoints into a
 * single flow rule with a mask. This only applies per-port; it does not try
 * to identify subscriptions to multiple ports on the same address.
 *
 * @pre The endpoints are IPv4 multicast addresses.
 */
std::vector<ibv_flow_t> create_flows(
    const ibv_qp_t &qp, const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    int port_num);

#if SPEAD2_USE_IBV_MPRQ

class ibv_exp_query_intf_error_category : public std::error_category
{
public:
    virtual const char *name() const noexcept override;
    virtual std::string message(int condition) const override;
    virtual std::error_condition default_error_condition(int condition) const noexcept override;
};

std::error_category &ibv_exp_query_intf_category();

class ibv_exp_cq_family_v1_t : public std::unique_ptr<ibv_exp_cq_family_v1, detail::ibv_intf_deleter>
{
public:
    ibv_exp_cq_family_v1_t() = default;
    ibv_exp_cq_family_v1_t(const rdma_cm_id_t &cm_id, const ibv_cq_t &cq);
};

class ibv_exp_wq_t : public std::unique_ptr<ibv_exp_wq, detail::ibv_exp_wq_deleter>
{
public:
    ibv_exp_wq_t() = default;
    ibv_exp_wq_t(const rdma_cm_id_t &cm_id, ibv_exp_wq_init_attr *attr);

    void modify(ibv_exp_wq_state state);
};

class ibv_exp_wq_family_t : public std::unique_ptr<ibv_exp_wq_family, detail::ibv_intf_deleter>
{
public:
    ibv_exp_wq_family_t() = default;
    ibv_exp_wq_family_t(const rdma_cm_id_t &cm_id, const ibv_exp_wq_t &wq);
};

class ibv_exp_rwq_ind_table_t : public std::unique_ptr<ibv_exp_rwq_ind_table, detail::ibv_exp_rwq_ind_table_deleter>
{
public:
    ibv_exp_rwq_ind_table_t() = default;
    ibv_exp_rwq_ind_table_t(const rdma_cm_id_t &cm_id, ibv_exp_rwq_ind_table_init_attr *attr);
};

/// Construct a table with a single entry
ibv_exp_rwq_ind_table_t create_rwq_ind_table(
    const rdma_cm_id_t &cm_id, const ibv_pd_t &pd, const ibv_exp_wq_t &wq);

class ibv_exp_res_domain_t : public std::unique_ptr<ibv_exp_res_domain, detail::ibv_exp_res_domain_deleter>
{
public:
    ibv_exp_res_domain_t() = default;
    ibv_exp_res_domain_t(const rdma_cm_id_t &cm_id, ibv_exp_res_domain_init_attr *attr);
};

#endif // SPEAD2_USE_IBV_MPRQ

} // namespace spead2

#endif // SPEAD2_USE_IBV

#endif // SPEAD2_COMMON_IBV_H
