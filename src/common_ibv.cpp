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

#include <spead2/common_features.h>
#if SPEAD2_USE_IBV
#include <cerrno>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <atomic>
#include <algorithm>
#include <system_error>
#include <boost/asio.hpp>
#include <spead2/common_logging.h>
#include <spead2/common_ibv.h>
#include <spead2/common_semaphore.h>
#include <spead2/common_endian.h>
#include <spead2/common_raw_packet.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

namespace spead2
{

namespace detail
{

/* While compilation requires a relatively modern rdma-core, at runtime we
 * may be linked to an old version. The ABI for ibv_create_flow and
 * ibv_destroy_flow changed at around v12 or v13.
 *
 * We can work around it by mimicking the internals of verbs.h and
 * selecting the correct slot.
 */
struct verbs_context_fixed
{
    int (*ibv_destroy_flow)(ibv_flow *flow);
    int (*old_ibv_destroy_flow)(ibv_flow *flow);
    ibv_flow * (*ibv_create_flow)(ibv_qp *qp, ibv_flow_attr *flow_attr);
    ibv_flow * (*old_ibv_create_flow)(ibv_qp *qp, ibv_flow_attr *flow_attr);
    void (*padding[6])(void);
    std::uint64_t has_comp_mask;
    std::size_t sz;
    ibv_context context;
};

// Returns the wrapping verbs_context_fixed if it seems like there is one,
// otherwise NULL.
static const verbs_context_fixed *get_verbs_context_fixed(ibv_context *ctx)
{
    if (!ctx || ctx->abi_compat != __VERBS_ABI_IS_EXTENDED)
        return NULL;
    const verbs_context_fixed *vctx = (const verbs_context_fixed *)(
        (const char *)(ctx) - offsetof(verbs_context_fixed, context));
    if (vctx->sz >= sizeof(*vctx))
        return vctx;
    else
        return NULL;
}

static ibv_flow *wrap_ibv_create_flow(ibv_qp *qp, ibv_flow_attr *flow_attr)
{
    errno = 0;
    ibv_flow *flow = ibv_create_flow(qp, flow_attr);
    if (!flow && (errno == 0 || errno == EOPNOTSUPP))
    {
        const verbs_context_fixed *vctx = get_verbs_context_fixed(qp->context);
        if (vctx->old_ibv_create_flow && !vctx->ibv_create_flow)
            flow = vctx->old_ibv_create_flow(qp, flow_attr);
        else if (errno == 0)
            errno = EOPNOTSUPP;  // old versions of ibv_create_flow neglect to set errno
    }
    return flow;
}

static int wrap_ibv_destroy_flow(ibv_flow *flow)
{
    errno = 0;
    int result = ibv_destroy_flow(flow);
    /* While ibv_destroy_flow is supposed to return an errno on failure, the
     * header files have in the past returned negated error numbers.
     */
    if (result != 0)
    {
        if (std::abs(result) == ENOSYS || std::abs(result) == EOPNOTSUPP)
        {
            const verbs_context_fixed *vctx = get_verbs_context_fixed(flow->context);
            if (vctx->old_ibv_destroy_flow && !vctx->ibv_destroy_flow)
                result = vctx->old_ibv_destroy_flow(flow);
        }
    }
    return result;
}

void ibv_flow_deleter::operator()(ibv_flow *flow)
{
    wrap_ibv_destroy_flow(flow);
}

} // namespace detail

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

ibv_device_attr rdma_cm_id_t::query_device() const
{
    assert(get());
    ibv_device_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    int status = ibv_query_device(get()->verbs, &attr);
    if (status != 0)
        throw_errno("ibv_query_device failed", status);
    return attr;
}

ibv_device_attr_ex rdma_cm_id_t::query_device_ex(const struct ibv_query_device_ex_input *input) const
{
    assert(get());
    ibv_device_attr_ex attr;
    ibv_query_device_ex_input dummy_input;
    if (!input)
    {
        std::memset(&dummy_input, 0, sizeof(dummy_input));
        input = &dummy_input;
    }
    std::memset(&attr, 0, sizeof(attr));
    int status = ibv_query_device_ex(get()->verbs, input, &attr);
    if (status != 0)
        throw_errno("ibv_query_device_ex failed", status);
    return attr;
}

#if SPEAD2_USE_MLX5DV
bool rdma_cm_id_t::mlx5dv_is_supported() const
{
    assert(get());
    try
    {
        return spead2::mlx5dv_is_supported(get()->verbs->device);
    }
    catch (std::system_error &)
    {
        return false;
    }
}

mlx5dv_context rdma_cm_id_t::mlx5dv_query_device() const
{
    assert(get());
    mlx5dv_context attr;
    std::memset(&attr, 0, sizeof(attr));
    // TODO: set other flags if they're defined (will require configure-time
    // detection).
    attr.comp_mask = MLX5DV_CONTEXT_MASK_STRIDING_RQ | MLX5DV_CONTEXT_MASK_CLOCK_INFO_UPDATE;
    int status = spead2::mlx5dv_query_device(get()->verbs, &attr);
    if (status != 0)
        throw_errno("mlx5dv_query_device failed", status);
    return attr;
}
#endif  // SPEAD2_USE_MLX5DV

ibv_context_t::ibv_context_t(struct ibv_device *device)
{
    ibv_context *ctx = ibv_open_device(device);
    if (!ctx)
        throw_errno("ibv_open_device failed");
    reset(ctx);
}

ibv_context_t::ibv_context_t(const boost::asio::ip::address &addr)
{
    /* Use rdma_cm_id_t to get an existing device context, then
     * query it for its GUID and find the corresponding device.
     */
    rdma_event_channel_t event_channel;
    rdma_cm_id_t cm_id(event_channel, nullptr, RDMA_PS_UDP);
    cm_id.bind_addr(addr);
    ibv_device_attr attr = cm_id.query_device();

    struct ibv_device **devices;
    devices = ibv_get_device_list(nullptr);
    if (devices == nullptr)
        throw_errno("ibv_get_device_list failed");

    ibv_device *device = nullptr;
    for (ibv_device **d = devices; *d != nullptr; d++)
        if (ibv_get_device_guid(*d) == attr.node_guid)
        {
            device = *d;
            break;
        }
    if (device == nullptr)
    {
        ibv_free_device_list(devices);
        throw_errno("no matching device found", ENOENT);
    }

    ibv_context *ctx = ibv_open_device(device);
    if (!ctx)
    {
        ibv_free_device_list(devices);
        throw_errno("ibv_open_device failed");
    }
    reset(ctx);
    ibv_free_device_list(devices);
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
    return wrap_fd(io_service, get()->fd);
}

bool ibv_comp_channel_t::get_event(ibv_cq **cq, void **context)
{
    assert(get());
    errno = 0;
    int status = ibv_get_cq_event(get(), cq, context);
    if (status < 0)
    {
        if (errno == EAGAIN)
            return false;
        else
            throw_errno("ibv_get_cq_event failed");
    }
    return true;
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

ibv_cq_ex_t::ibv_cq_ex_t(const rdma_cm_id_t &cm_id, ibv_cq_init_attr_ex *cq_attr)
{
    errno = 0;
    ibv_cq_ex *cq = ibv_create_cq_ex(cm_id->verbs, cq_attr);
    if (!cq)
        throw_errno("ibv_create_cq_ex failed");
    reset(ibv_cq_ex_to_cq(cq));
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
    {
        if (errno == EINVAL && init_attr->qp_type == IBV_QPT_RAW_PACKET)
            throw_errno(
                "ibv_create_qp failed (could be a permission problem - do you have CAP_NET_RAW?)");
        else
            throw_errno("ibv_create_qp failed");
    }
    reset(qp);
}

ibv_qp_t::ibv_qp_t(const rdma_cm_id_t &cm_id, ibv_qp_init_attr_ex *init_attr)
{
    errno = 0;
    ibv_qp *qp = ibv_create_qp_ex(cm_id->verbs, init_attr);
    if (!qp)
    {
        if (errno == EINVAL && init_attr->qp_type == IBV_QPT_RAW_PACKET)
            throw_errno(
                "ibv_create_qp_ex failed (could be a permission problem - do you have CAP_NET_RAW?)");
        else
            throw_errno("ibv_create_qp_ex failed");
    }
    reset(qp);
}

ibv_mr_t::ibv_mr_t(const ibv_pd_t &pd, void *addr, std::size_t length, int access,
                   bool allow_relaxed_ordering)
{
#ifndef IBV_ACCESS_RELAXED_ORDERING
    const int IBV_ACCESS_OPTIONAL_RANGE = 0x3ff00000;
    const int IBV_ACCESS_RELAXED_ORDERING = 1 << 20;
#endif
    if (allow_relaxed_ordering)
        access |= IBV_ACCESS_RELAXED_ORDERING;
    /* Emulate the ibv_reg_mr macro in verbs.h. If access contains optional
     * flags, we have to call ibv_reg_mr_iova2 rather than the ibv_reg_mr
     * symbol. If the function is not available, just mask out the bits,
     * which is what ibv_reg_mr_iova2 does if the kernel doesn't support
     * them.
     */
    errno = 0;
    ibv_mr *mr;
    if (!(access & IBV_ACCESS_OPTIONAL_RANGE))
        mr = ibv_reg_mr(pd.get(), addr, length, access);
    else if (!has_ibv_reg_mr_iova2())
        mr = ibv_reg_mr(pd.get(), addr, length, access & ~IBV_ACCESS_OPTIONAL_RANGE);
    else
        mr = ibv_reg_mr_iova2(pd.get(), addr, length, std::uintptr_t(addr), access);
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
    ibv_flow *flow = detail::wrap_ibv_create_flow(qp.get(), flow_attr);
    if (!flow)
        throw_errno("ibv_create_flow failed");
    reset(flow);
}

ibv_flow_t create_flow(
    const ibv_qp_t &qp, const boost::asio::ip::udp::endpoint &endpoint,
    int port_num)
{
    struct
    {
        ibv_flow_attr attr;
        ibv_flow_spec_eth eth __attribute__((packed));
        ibv_flow_spec_ipv4 ip __attribute__((packed));
        ibv_flow_spec_tcp_udp udp __attribute__((packed));
    } flow_rule;
    std::memset(&flow_rule, 0, sizeof(flow_rule));

    flow_rule.attr.type = IBV_FLOW_ATTR_NORMAL;
    flow_rule.attr.priority = 0;
    flow_rule.attr.size = sizeof(flow_rule);
    flow_rule.attr.num_of_specs = 3;
    flow_rule.attr.port = port_num;

    flow_rule.eth.type = IBV_FLOW_SPEC_ETH;
    flow_rule.eth.size = sizeof(flow_rule.eth);
    flow_rule.ip.type = IBV_FLOW_SPEC_IPV4;
    flow_rule.ip.size = sizeof(flow_rule.ip);

    if (!endpoint.address().is_unspecified())
    {
        /* At least the ConnectX-3 cards seem to require an Ethernet match. We
         * thus have to construct the either the MAC address corresponding to
         * the IP multicast address from RFC 7042, the interface address for
         * unicast.
         */
        mac_address dst_mac;
        if (endpoint.address().is_multicast())
            dst_mac = multicast_mac(endpoint.address());
        else
            dst_mac = interface_mac(endpoint.address());
        std::memcpy(&flow_rule.eth.val.dst_mac, &dst_mac, sizeof(dst_mac));
        std::memset(&flow_rule.eth.mask.dst_mac, 0xFF, sizeof(flow_rule.eth.mask.dst_mac));

        auto bytes = endpoint.address().to_v4().to_bytes(); // big-endian address
        std::memcpy(&flow_rule.ip.val.dst_ip, &bytes, sizeof(bytes));
        std::memset(&flow_rule.ip.mask.dst_ip, 0xFF, sizeof(flow_rule.ip.mask.dst_ip));
    }

    flow_rule.udp.type = IBV_FLOW_SPEC_UDP;
    flow_rule.udp.size = sizeof(flow_rule.udp);
    flow_rule.udp.val.dst_port = htobe16(endpoint.port());
    flow_rule.udp.mask.dst_port = 0xFFFF;

    return ibv_flow_t(qp, &flow_rule.attr);
}

std::vector<ibv_flow_t> create_flows(
    const ibv_qp_t &qp,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    int port_num)
{
    /* Note: some NICs support flow rules with non-trivial masks. However,
     * using such rules can lead to subtle problems when there are multiple
     * receivers on the same NIC subscribing to common groups. See #66 for
     * more details.
     */
    std::vector<ibv_flow_t> flows;
    for (const auto &ep : endpoints)
        flows.push_back(create_flow(qp, ep, port_num));
    return flows;
}

ibv_wq_t::ibv_wq_t(const rdma_cm_id_t &cm_id, ibv_wq_init_attr *attr)
{
    ibv_wq *wq = ibv_create_wq(cm_id->verbs, attr);
    if (!wq)
        throw_errno("ibv_create_wq failed");
    reset(wq);
}

void ibv_wq_t::modify(ibv_wq_state state)
{
    ibv_wq_attr wq_attr;
    std::memset(&wq_attr, 0, sizeof(wq_attr));
    wq_attr.wq_state = state;
    wq_attr.attr_mask = IBV_WQ_ATTR_STATE;
    int status = ibv_modify_wq(get(), &wq_attr);
    if (status != 0)
        throw_errno("ibv_modify_wq failed", status);
}

#if SPEAD2_USE_MLX5DV
ibv_wq_mprq_t::ibv_wq_mprq_t(const rdma_cm_id_t &cm_id, ibv_wq_init_attr *attr, mlx5dv_wq_init_attr *mlx5_attr)
    : stride_size(1U << mlx5_attr->striding_rq_attrs.single_stride_log_num_of_bytes),
    n_strides(1U << mlx5_attr->striding_rq_attrs.single_wqe_log_num_of_strides),
    data_offset(mlx5_attr->striding_rq_attrs.two_byte_shift_en ? 2 : 0)
{
    assert(mlx5_attr->comp_mask & MLX5DV_WQ_INIT_ATTR_MASK_STRIDING_RQ);
    ibv_wq *wq = mlx5dv_create_wq(cm_id->verbs, attr, mlx5_attr);
    if (!wq)
        throw_errno("mlx5dv_create_wq failed");
    mlx5dv_obj obj;
    obj.rwq.in = wq;
    obj.rwq.out = &rwq;
    int ret = mlx5dv_init_obj(&obj, MLX5DV_OBJ_RWQ);
    if (ret != 0)
    {
        ibv_destroy_wq(wq);
        throw_errno("mlx5dv_init_obj failed", ret);
    }
    if (rwq.stride != sizeof(mlx5_mprq_wqe))
    {
        ibv_destroy_wq(wq);
        throw_errno("multi-packet receive queue has unexpected stride", EOPNOTSUPP);
    }
    reset(wq);
}

void ibv_wq_mprq_t::post_recv(ibv_sge *sge)
{
    if (head - tail >= rwq.wqe_cnt)
        throw_errno("Multi-packet receive queue is full", ENOMEM);

    int ind = head & (rwq.wqe_cnt - 1);
    mlx5_mprq_wqe *wqe = (mlx5_mprq_wqe *) rwq.buf + ind;
    memset(&wqe->nseg, 0, sizeof(wqe->nseg));
    wqe->dseg.byte_count = htobe32(sge->length);
    wqe->dseg.lkey = htobe32(sge->lkey);
    wqe->dseg.addr = htobe64(sge->addr);
    head++;
    /* Update the doorbell to tell the HW about the new entry. This must
     * be ordered after the writes to the buffer, so we hope that any
     * sensible platform will make std::atomic_uint32_t just a wrapper
     * around a uint32_t.
     */
    static_assert(sizeof(std::atomic_uint32_t) == sizeof(std::uint32_t),
                  "std::atomic_uint32_t has the wrong size");
    std::atomic<std::uint32_t> *dbrec = reinterpret_cast<std::atomic<std::uint32_t> *>(rwq.dbrec);
    dbrec->store(htobe32(head & 0xffff), std::memory_order_release);
}

void ibv_wq_mprq_t::read_wc(const ibv_cq_ex_t &cq, std::uint32_t &byte_len,
                            std::uint32_t &offset, int &flags) noexcept
{
    /* This is actually a packed field: lower 16 bytes are the byte count,
     * top bit is the "filler" flag, remaining bits are the number of
     * strides consumed.
     */
    std::uint32_t byte_cnt = ibv_wc_read_byte_len(cq.get());
    std::uint32_t strides = byte_cnt >> 16;
    byte_len = (byte_cnt & 0xffff) - data_offset;
    offset = tail_strides * stride_size + data_offset;
    flags = 0;
    if (strides & 0x8000)
    {
        strides -= 0x8000;
        flags |= FLAG_FILLER;
    }
    tail_strides += strides;
    if (tail_strides >= n_strides)
    {
        assert(tail_strides <= n_strides);
        flags |= FLAG_LAST;
        tail++;
        tail_strides = 0;
    }
}

#endif // SPEAD2_USE_MLX5DV

ibv_rwq_ind_table_t::ibv_rwq_ind_table_t(const rdma_cm_id_t &cm_id, ibv_rwq_ind_table_init_attr *attr)
{
    ibv_rwq_ind_table *table = ibv_create_rwq_ind_table(cm_id->verbs, attr);
    if (!table)
        throw_errno("ibv_create_rwq_ind_table failed");
    reset(table);
}

ibv_rwq_ind_table_t create_rwq_ind_table(const rdma_cm_id_t &cm_id, const ibv_wq_t &wq)
{
    ibv_rwq_ind_table_init_attr attr;
    ibv_wq *tbl[1] = {wq.get()};
    std::memset(&attr, 0, sizeof(attr));
    attr.log_ind_tbl_size = 0;
    attr.ind_tbl = tbl;
    return ibv_rwq_ind_table_t(cm_id, &attr);
}

} // namespace spead

#endif // SPEAD2_USE_IBV
