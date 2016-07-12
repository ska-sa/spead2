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

#include <spead2/common_features.h>
#if SPEAD2_USE_IBV
#include <cerrno>
#include <cstring>
#include <cassert>
#include <memory>
#include <boost/asio.hpp>
#include <spead2/common_logging.h>
#include <spead2/common_ibv.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

namespace spead2
{

rdma_event_channel_t::rdma_event_channel_t()
{
    errno = 0;
    rdma_event_channel *event_channel = rdma_create_event_channel();
    if (!event_channel)
        throw_errno("rdma_create_event_channel failed");
    reset(event_channel);
}

rdma_cm_id_t::rdma_cm_id_t(const rdma_event_channel_t &event_channel, void *context, rdma_port_space ps)
{
    rdma_cm_id *cm_id = nullptr;
    errno = 0;
    int status = rdma_create_id(event_channel.get(), &cm_id, context, ps);
    if (status < 0)
        throw_errno("rdma_create_id failed");
    reset(cm_id);
}

void rdma_cm_id_t::bind_addr(const boost::asio::ip::address &addr)
{
    assert(get());
    boost::asio::ip::udp::endpoint endpoint(addr, 0);
    errno = 0;
    int status = rdma_bind_addr(get(), endpoint.data());
    if (status < 0)
        throw_errno("rdma_bind_addr failed");
    if (get()->verbs == nullptr)
        throw_errno("rdma_bind_addr did not bind to an RDMA device", ENODEV);
}

ibv_comp_channel_t::ibv_comp_channel_t(const rdma_cm_id_t &cm_id)
{
    errno = 0;
    ibv_comp_channel *comp_channel = ibv_create_comp_channel(cm_id->verbs);
    if (!comp_channel)
        throw_errno("ibv_create_comp_channel failed");
    reset(comp_channel);
}

boost::asio::posix::stream_descriptor ibv_comp_channel_t::wrap(
    boost::asio::io_service &io_service) const
{
    assert(get());
    int fd = dup(get()->fd);
    if (fd < 0)
        throw_errno("dup failed");
    boost::asio::posix::stream_descriptor descriptor(io_service, fd);
    descriptor.native_non_blocking(true);
    return descriptor;
}

void ibv_comp_channel_t::get_event(ibv_cq **cq, void **context)
{
    assert(get());
    errno = 0;
    int status = ibv_get_cq_event(get(), cq, context);
    if (status < 0)
        throw_errno("ibv_get_cq_event failed");
}

ibv_cq_t::ibv_cq_t(
    const rdma_cm_id_t &cm_id, int cqe, void *context,
    const ibv_comp_channel_t &comp_channel, int comp_vector)
{
    errno = 0;
    ibv_cq *cq = ibv_create_cq(cm_id->verbs, cqe, context, comp_channel.get(), comp_vector);
    if (!cq)
        throw_errno("ibv_create_cq failed");
    reset(cq);
}

ibv_cq_t::ibv_cq_t(const rdma_cm_id_t &cm_id, int cqe, void *context)
{
    errno = 0;
    ibv_cq *cq = ibv_create_cq(cm_id->verbs, cqe, context, nullptr, 0);
    if (!cq)
        throw_errno("ibv_create_cq failed");
    reset(cq);
}

void ibv_cq_t::req_notify(bool solicited_only)
{
    assert(get());
    int status = ibv_req_notify_cq(get(), int(solicited_only));
    if (status != 0)
        throw_errno("ibv_req_notify_cq failed", status);
}

int ibv_cq_t::poll(int num_entries, ibv_wc *wc)
{
    assert(get());
    int received = ibv_poll_cq(get(), num_entries, wc);
    if (received < 0)
        throw_errno("ibv_poll_cq failed");
    return received;
}

void ibv_cq_t::ack_events(unsigned int nevents)
{
    assert(get());
    ibv_ack_cq_events(get(), nevents);
}

ibv_pd_t::ibv_pd_t(const rdma_cm_id_t &cm_id)
{
    errno = 0;
    ibv_pd *pd = ibv_alloc_pd(cm_id->verbs);
    if (!pd)
        throw_errno("ibv_alloc_pd failed");
    reset(pd);
}

ibv_qp_t::ibv_qp_t(const ibv_pd_t &pd, ibv_qp_init_attr *init_attr)
{
    errno = 0;
    ibv_qp *qp = ibv_create_qp(pd.get(), init_attr);
    if (!qp)
        throw_errno("ibv_create_qp failed");
    reset(qp);
}

ibv_mr_t::ibv_mr_t(const ibv_pd_t &pd, void *addr, std::size_t length, int access)
{
    errno = 0;
    ibv_mr * mr = ibv_reg_mr(pd.get(), addr, length, IBV_ACCESS_LOCAL_WRITE);
    if (!mr)
        throw_errno("ibv_reg_mr failed");
    reset(mr);
}

void ibv_qp_t::modify(ibv_qp_attr *attr, int attr_mask)
{
    assert(get());
    int status = ibv_modify_qp(get(), attr, attr_mask);
    if (status != 0)
        throw_errno("ibv_modify_qp failed", status);
}

void ibv_qp_t::modify(ibv_qp_state qp_state)
{
    ibv_qp_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = qp_state;
    modify(&attr, IBV_QP_STATE);
}

void ibv_qp_t::modify(ibv_qp_state qp_state, int port_num)
{
    ibv_qp_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = qp_state;
    attr.port_num = port_num;
    modify(&attr, IBV_QP_STATE | IBV_QP_PORT);
}

void ibv_qp_t::post_recv(ibv_recv_wr *wr)
{
    assert(get());
    ibv_recv_wr *bad_wr;
    int status = ibv_post_recv(get(), wr, &bad_wr);
    if (status != 0)
        throw_errno("ibv_post_recv failed", status);
}

void ibv_qp_t::post_send(ibv_send_wr *wr)
{
    assert(get());
    ibv_send_wr *bad_wr;
    int status = ibv_post_send(get(), wr, &bad_wr);
    if (status != 0)
        throw_errno("ibv_post_send failed", status);
}

ibv_flow_t::ibv_flow_t(const ibv_qp_t &qp, ibv_flow_attr *flow_attr)
{
    errno = 0;
    ibv_flow *flow = ibv_create_flow(qp.get(), flow_attr);
    if (!flow)
        throw_errno("ibv_create_flow failed");
    reset(flow);
}

} // namespace spead

#endif // SPEAD2_USE_IBV
