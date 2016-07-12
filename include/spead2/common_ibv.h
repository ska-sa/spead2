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

#ifndef SPEAD2_COMMON_IBV_H
#define SPEAD2_COMMON_IBV_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>
#include <memory>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <boost/asio.hpp>

#if SPEAD2_USE_IBV

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

    void req_notify(bool solicited_only);
    int poll(int num_entries, ibv_wc *wc);
    void ack_events(unsigned int nevents);
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

} // namespace spead2

#endif // SPEAD2_USE_IBV

#endif // SPEAD2_COMMON_IBV_H
