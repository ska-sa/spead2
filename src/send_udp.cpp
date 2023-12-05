/* Copyright 2015, 2019-2020, 2023 National Research Foundation (SARAO)
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

#include <cstddef>
#include <cstring>
#include <utility>
#include <algorithm>
#include <boost/asio.hpp>
#include <spead2/send_udp.h>
#include <spead2/send_writer.h>
#include <spead2/common_defines.h>
#include <spead2/common_socket.h>
#if SPEAD2_USE_SENDMMSG
# include <sys/types.h>
# include <sys/socket.h>
# include <netinet/in.h>
# include <netinet/udp.h>
#endif
#if SPEAD2_USE_MSG_ZEROCOPY
# include <linux/errqueue.h>
#endif

namespace spead2::send
{

namespace
{

class udp_writer : public writer
{
private:
    boost::asio::ip::udp::socket socket;
    std::vector<boost::asio::ip::udp::endpoint> endpoints;

    virtual void wakeup() override final;

    /* NB: Linux has a maximum of 64 segments for GSO (UDP_MAX_SEGMENTS in the
     * kernel, but it doesn't seem to be exposed to userspace). If max_batch
     * is increased, logic will need to be added to the GSO merging to prevent
     * creating messages bigger than this.
     */
    static constexpr int max_batch = 64;
#if SPEAD2_USE_SENDMMSG
    /**
     * Maximum number of packets that can be in-flight. Unlike
     * @ref match_batch, this includes packets for which the sendmmsg call
     * completed but which we still need to retain due to zero copy. It must
     * be larger than @ref max_batch and must be a power of two.
     */
    static constexpr int max_outstanding = 256;

    // Some magic values for current_gso_size
    /// GSO allowed, but socket option not currently set
    [[maybe_unused]] static constexpr int gso_inactive = 0;
    /// GSO failed; do not try again
    [[maybe_unused]] static constexpr int gso_disabled = -1;
    /// Last send with GSO failed; retrying without GSO
    [[maybe_unused]] static constexpr int gso_probe = -2;

    static constexpr int max_gso_message_size = 65535;  // maximum size the kernel will accept
    struct mmsghdr msgvec[max_outstanding];
    std::vector<struct iovec> msg_iov;
    struct
    {
        transmit_packet packet;
        std::unique_ptr<std::uint8_t[]> scratch;
        bool merged; // packet is part of the same message as the previous packet
        bool zerocopy;  // packet was zero-copied and we need to wait for notification
        std::uint32_t seq;  // kernel sequence number for zero-copy notifications
    } packets[max_outstanding];
    /// Oldest packet for which we are awaiting completion acknowledgement
    unsigned int head_packet = 0;
    /// Empty packet slot for adding next new packet
    unsigned int tail_packet = 0;
    int current_gso_size = gso_inactive;
    bool zerocopy_enabled = false;
    std::uint32_t next_seq = 0;  // next sequence number for MSG_ZEROCOPY

#if SPEAD2_USE_GSO
    /// Set the generic segmentation offload socket option
    void set_gso_size(int size, boost::system::error_code &result);
#endif

    /**
     * Get the next packet slot, wrapping at @ref max_outstanding
     */
    static int next_packet(int packet_index)
    {
        static_assert((max_outstanding & (max_outstanding - 1)) == 0,
                      "max_outstanding must be a power of two");
        return (packet_index + 1) & (max_outstanding - 1);
    }

#if SPEAD2_USE_MSG_ZEROCOPY
    /// Set the @c SO_ZEROCOPY socket option
    void enable_zerocopy(boost::system::error_code &result);

    /// Number of free entries in @ref packets
    int free_slots() const
    {
        /* Note: head_packet is normally less than tail_packet (except when
         * the data wraps around), but the unsigned overflow is well defined
         * and the bitmask will do the right thing.
         */
        if (head_packet == tail_packet)
            return max_outstanding;
        else
            return (head_packet - tail_packet) & (max_outstanding - 1);
    }

    /// Process any zerocopy notifications from the error queue.
    void reap();

    /// Request a call to @ref wakeup when there is another completion notification
    void wait_for_space();
#endif

    /**
     * Set up @ref msgvec from @ref msg_iov.
     *
     * The packets in [first_packet, last_packet) are assumed to have already
     * been set in @ref msg_iov, starting from @a first_iov. If @a gso_size is
     * positive, then multiple packets may be concatenated into a single
     * element of @ref msgvec, provided that all but the last have size
     * @a gso_size. Otherwise, each packet gets its own entry in @ref msgvec.
     *
     * Note that the packet indices wrap around at @ref max_outstanding.
     *
     * @return The past-the-end index into @ref msgvec after the packets are
     * filled in.
     */
    int prepare_msgvec(int first_packet, int last_packet, int first_msg, int first_iov, int gso_size);
    void send_packets(int first_packet, int last_packet, int first_msg, int last_msg);
#else
    std::unique_ptr<std::uint8_t[]> scratch;
#endif

public:
    udp_writer(
        io_service_ref io_service,
        boost::asio::ip::udp::socket &&socket,
        const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
        const stream_config &config,
        std::size_t buffer_size);

    virtual std::size_t get_num_substreams() const override final { return endpoints.size(); }
};

#if SPEAD2_USE_SENDMMSG

#if SPEAD2_USE_GSO
void udp_writer::set_gso_size(int size, boost::system::error_code &result)
{
    if (setsockopt(socket.native_handle(), IPPROTO_UDP, UDP_SEGMENT,
                   &size, sizeof(size)) == -1)
    {
        result.assign(errno, boost::asio::error::get_system_category());
    }
    else
    {
        result.clear();
    }
}
#endif

/**
 * Call sendmmsg. If it was called with MSG_ZEROCOPY and failed with ENOBUFS,
 * modify @a flags and call it again without zerocopy.
 */
static int sendmmsg_maybe_zerocopy(
    int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int &flags
)
{
    int ret = sendmmsg(sockfd, msgvec, vlen, flags);
#if SPEAD2_USE_MSG_ZEROCOPY
    if (ret == -1 && errno == ENOBUFS && (flags & MSG_ZEROCOPY))
    {
        flags &= ~MSG_ZEROCOPY;
        ret = sendmmsg(sockfd, msgvec, vlen, flags);
    }
#endif
    return ret;
}

void udp_writer::send_packets(int first_packet, int last_packet, int first_msg, int last_msg)
{
#if SPEAD2_USE_GSO
restart:
#endif
    int groups = 0;
    boost::system::error_code result;

    // Try sending
    int flags = MSG_DONTWAIT;
#if SPEAD2_USE_MSG_ZEROCOPY
    if (zerocopy_enabled)
        flags |= MSG_ZEROCOPY;
#endif
    int sent = sendmmsg_maybe_zerocopy(socket.native_handle(), msgvec + first_msg, last_msg - first_msg, flags);
    if (sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
    {
        /* Not all device drivers support GSO. If we were trying with GSO, try again
         * without.
         */
        result.assign(errno, boost::asio::error::get_system_category());
#if SPEAD2_USE_GSO
        if (current_gso_size == gso_probe)
        {
            /* We tried sending with GSO and it failed, but resending without GSO
             * also failed, so the fault is probably not lack of GSO support. Allow
             * GSO to be used again.
             */
            current_gso_size = gso_inactive;
        }
        else if (current_gso_size > 0)
        {
            set_gso_size(0, result);
            if (!result)
            {
                /* Re-compute msgvec without GSO */
                current_gso_size = gso_probe;
                last_msg = prepare_msgvec(first_packet, last_packet, first_msg,
                                          msgvec[first_msg].msg_hdr.msg_iov - msg_iov.data(),
                                          0);
                goto restart;
            }
        }
#endif
        do
        {
            auto *item = packets[first_packet].packet.item;
            if (!item->result)
                item->result = result;
            groups += packets[first_packet].packet.last;
            first_packet = next_packet(first_packet);
        } while (first_packet != last_packet && packets[first_packet].merged);
        first_msg++;
    }
    else if (sent > 0)
    {
        if (current_gso_size == gso_probe)
        {
            log_debug("disabling GSO because sending with it failed and without succeeded");
            // Sending with GSO failed and without GSO succeeded. The network
            // device probably does not support it, so don't try again.
            current_gso_size = gso_disabled;
        }
#if SPEAD2_USE_MSG_ZEROCOPY
        if (flags & MSG_ZEROCOPY)
        {
            for (int i = 0; i < sent; i++)
            {
                do
                {
                    packets[first_packet].zerocopy = true;
                    packets[first_packet].seq = next_seq;
                    first_packet = next_packet(first_packet);
                } while (first_packet != last_packet && packets[first_packet].merged);
                next_seq++;
            }
        }
        else
#endif
        {
            for (int i = 0; i < sent; i++)
            {
                do
                {
                    auto *item = packets[first_packet].packet.item;
                    item->bytes_sent += packets[first_packet].packet.size;
                    groups += packets[first_packet].packet.last;
                    first_packet = next_packet(first_packet);
                } while (first_packet != last_packet && packets[first_packet].merged);
            }
        }
        first_msg += sent;
    }

    if (groups > 0)
        groups_completed(groups);
    if (first_msg < last_msg)
    {
        // We didn't manage to send it all: schedule a new attempt once there is
        // buffer space.
        socket.async_wait(
            socket.wait_write,
            [this, first_packet, last_packet, first_msg, last_msg](
                const boost::system::error_code &
            ) {
                send_packets(first_packet, last_packet, first_msg, last_msg);
            });
    }
    else
    {
        post_wakeup();
    }
}

int udp_writer::prepare_msgvec(int first_packet, int last_packet, int first_msg, int first_iov, int gso_size)
{
    int merged_size = 0;
    int iov = first_iov;
    int msg = first_msg;
    auto prev_packet = &packets[first_packet];
    for (int i = first_packet; i != last_packet; i = next_packet(i))
    {
        /* Check if we can merge with the previous packet using generic
         * segmentation offload. */
        if (!SPEAD2_USE_GSO
            || i == first_packet
            || (int) prev_packet->packet.size != gso_size
            || packets[i].packet.substream_index != prev_packet->packet.substream_index
            || merged_size + packets[i].packet.size > max_gso_message_size)
        {
            // Can't merge, so initialise a new header
            auto &hdr = msgvec[msg].msg_hdr;
            hdr.msg_iov = &msg_iov[iov];
            hdr.msg_iovlen = 0;
            const auto &endpoint = endpoints[packets[i].packet.substream_index];
            hdr.msg_name = (void *) endpoint.data();
            hdr.msg_namelen = endpoint.size();
            msg++;
            packets[i].merged = false;
            packets[i].zerocopy = false;  // until we know otherwise
            merged_size = 0;
        }
        else
        {
            packets[i].merged = true;
        }
        auto &hdr = msgvec[msg - 1].msg_hdr;
        hdr.msg_iovlen += packets[i].packet.buffers.size();
        merged_size += packets[i].packet.size;
        iov += packets[i].packet.buffers.size();
        prev_packet = &packets[i];
    }
    return msg;
}

#if SPEAD2_USE_MSG_ZEROCOPY
void udp_writer::enable_zerocopy(boost::system::error_code &result)
{
    int enable = 1;
    if (setsockopt(socket.native_handle(), SOL_SOCKET, SO_ZEROCOPY,
                   &enable, sizeof(enable)) == -1)
    {
        result.assign(errno, boost::asio::error::get_system_category());
    }
    else
    {
        result.clear();
    }
}

void udp_writer::reap()
{
    int groups = 0;
    while (true)
    {
        struct msghdr msg = {};
        char control[CMSG_SPACE(sizeof(struct sock_extended_err))];
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        int ret = recvmsg(socket.native_handle(), &msg, MSG_ERRQUEUE | MSG_DONTWAIT);
        if (ret == -1)
        {
            if (errno == EWOULDBLOCK || errno == EAGAIN)
            {
                break;
            }
            else
            {
                log_errno("Failed to poll error queue for zerocopy notifications: %1% (%2%)");
            }
        }
        else
        {
            for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg); cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg))
            {
                if ((cmsg->cmsg_level == SOL_IP && cmsg->cmsg_type == IP_RECVERR)
                    || (cmsg->cmsg_level == SOL_IPV6 && cmsg->cmsg_type == IPV6_RECVERR))
                {
                    struct sock_extended_err serr;
                    memcpy(&serr, CMSG_DATA(cmsg), sizeof(serr));
                    if (serr.ee_errno == 0 && serr.ee_origin == SO_EE_ORIGIN_ZEROCOPY)
                    {
                        /* TODO: the kernel doesn't guarantee in-order notification.
                         * If this range doesn't cover the head of the queue,
                         * we should mark other packets in the queue as no longer
                         * zerocopy packets.
                         */
                        uint32_t seq_first = serr.ee_info;
                        uint32_t seq_last = serr.ee_data; // NB: inclusive!
                        while (head_packet != tail_packet
                               && (!packets[head_packet].zerocopy
                                   || (seq_first <= packets[head_packet].seq
                                       && packets[head_packet].seq <= seq_last)))
                        {
                            auto *item = packets[head_packet].packet.item;
                            item->bytes_sent += packets[head_packet].packet.size;
                            groups += packets[head_packet].packet.last;
                            head_packet = next_packet(head_packet);
                        }
                    }
                    break;  // don't need to check any more cmsgs
                }
            }
        }
    }
    if (groups > 0)
        groups_completed(groups);
}

void udp_writer::wait_for_space()
{
    socket.async_wait(socket.wait_error, [this](const boost::system::error_code &) { wakeup(); });
}

#endif // SPEAD2_USE_MSG_ZEROCOPY

void udp_writer::wakeup()
{
#if SPEAD2_USE_MSG_ZEROCOPY
    reap();
    if (free_slots() < max_batch)
    {
        wait_for_space();
        return;
    }
#endif

    packet_result result = get_packet(packets[tail_packet].packet, packets[tail_packet].scratch.get());
    switch (result)
    {
    case packet_result::SLEEP:
        sleep();
        return;
    case packet_result::EMPTY:
        // If there are any outstanding packets, we need to reap them before we can
        // go completely idle.
#if SPEAD2_USE_MSG_ZEROCOPY
        if (head_packet != tail_packet)
        {
            wait_for_space();
            return;
        }
#endif
        request_wakeup();
        return;
    case packet_result::SUCCESS:
        break;
    }

    // We have at least one packet to send. See if we can get some more.
    int n;
    std::size_t n_iov = packets[tail_packet].packet.buffers.size();
    std::size_t max_size = packets[tail_packet].packet.size;
    int orig_tail_packet = tail_packet;
    tail_packet = next_packet(tail_packet);
    for (n = 1; n < max_batch; n++)
    {
        result = get_packet(packets[tail_packet].packet, packets[tail_packet].scratch.get());
        if (result != packet_result::SUCCESS)
            break;
        n_iov += packets[tail_packet].packet.buffers.size();
        max_size = std::max(max_size, packets[n].packet.size);
        tail_packet = next_packet(tail_packet);
    }

#if SPEAD2_USE_GSO
    int new_gso_size = max_size;
    if (new_gso_size != current_gso_size && current_gso_size >= 0)
    {
        boost::system::error_code result;
        set_gso_size(new_gso_size, result);
        if (!result)
            current_gso_size = new_gso_size;
        else if (result == boost::system::errc::no_protocol_option) // ENOPROTOOPT
        {
            /* Socket option is not supported on this platform. Just
             * disable GSO in our code.
             */
            log_debug("disabling GSO because socket option is not supported");
            current_gso_size = gso_disabled;
        }
        else
        {
            /* Something else has gone wrong. Make a best effort to disable
             * GSO on the socket.
             */
            log_warning("failed to set UDP_SEGMENT socket option to %1%: %2% (%3%)",
                        new_gso_size, result.value(), result.message());
            set_gso_size(0, result);
            if (!result)
                current_gso_size = 0;
        }
    }
#endif

    /* Fill in msg_iov from the packets */
    msg_iov.resize(n_iov);
    int iov = 0;
    int p = orig_tail_packet;
    for (int i = 0; i < n; i++)
    {
        for (const auto &buffer : packets[p].packet.buffers)
        {
            msg_iov[iov].iov_base = const_cast<void *>(
                boost::asio::buffer_cast<const void *>(buffer));
            msg_iov[iov].iov_len = boost::asio::buffer_size(buffer);
            iov++;
        }
        p = next_packet(p);
    }
    int n_msgs = prepare_msgvec(orig_tail_packet, tail_packet, 0, 0, current_gso_size);
    send_packets(orig_tail_packet, tail_packet, 0, n_msgs);
}

#else // SPEAD2_USE_SENDMMSG

void udp_writer::wakeup()
{
    for (int i = 0; i < max_batch; i++)
    {
        transmit_packet data;
        packet_result result = get_packet(data, scratch.get());
        switch (result)
        {
        case packet_result::SLEEP:
            sleep();
            return;
        case packet_result::EMPTY:
            request_wakeup();
            return;
        case packet_result::SUCCESS:
            break;
        }

        // First try a synchronous send
        auto *item = data.item;
        bool last = data.last;
        const auto &endpoint = endpoints[data.substream_index];
        boost::system::error_code ec;
        std::size_t bytes = socket.send_to(data.buffers, endpoint, 0, ec);
        if (ec == boost::asio::error::would_block)
        {
            // Socket buffer is full, so do an asynchronous send
            auto handler = [this, item, last](const boost::system::error_code &ec, std::size_t bytes_transferred)
            {
                item->bytes_sent += bytes_transferred;
                if (!item->result)
                    item->result = ec;
                if (last)
                    groups_completed(1);
                wakeup();
            };
            socket.async_send_to(data.buffers, endpoints[data.substream_index],
                                 std::move(handler));
            return;
        }
        else
        {
            item->bytes_sent += bytes;
            if (!item->result)
                item->result = ec;
            if (last)
                groups_completed(1);
        }
    }
    post_wakeup();
}

#endif // !SPEAD2_USE_SENDMMSG

udp_writer::udp_writer(
    io_service_ref io_service,
    boost::asio::ip::udp::socket &&socket,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size)
    : writer(std::move(io_service), config),
    socket(std::move(socket)),
    endpoints(endpoints)
#if !SPEAD2_USE_SENDMMSG
    , scratch(new std::uint8_t[config.get_max_packet_size()])
#endif
{
    if (!socket_uses_io_service(this->socket, get_io_service()))
        throw std::invalid_argument("I/O service does not match the socket's I/O service");
    auto protocol = this->socket.local_endpoint().protocol();
    for (const auto &endpoint : endpoints)
        if (endpoint.protocol() != protocol)
            throw std::invalid_argument("Endpoint does not match protocol of the socket");
    set_socket_send_buffer_size(this->socket, buffer_size);
    this->socket.non_blocking(true);
#if SPEAD2_USE_SENDMMSG
    std::memset(&msgvec, 0, sizeof(msgvec));
    for (int i = 0; i < max_outstanding; i++)
        packets[i].scratch.reset(new std::uint8_t[config.get_max_packet_size()]);
#if SPEAD2_USE_MSG_ZEROCOPY
    {
        boost::system::error_code result;
        enable_zerocopy(result);
        if (result)
            log_debug("Socket does not support zerocopy");
        else
            zerocopy_enabled = true;
    }
#endif
#endif
}

} // anonymous namespace

static boost::asio::ip::udp::socket make_socket(
    boost::asio::io_service &io_service,
    const boost::asio::ip::udp &protocol,
    const boost::asio::ip::address &interface_address)
{
    boost::asio::ip::udp::socket socket(io_service, protocol);
    if (!interface_address.is_unspecified())
        socket.bind(boost::asio::ip::udp::endpoint(interface_address, 0));
    return socket;
}

static boost::asio::ip::udp get_protocol(const std::vector<boost::asio::ip::udp::endpoint> &endpoints)
{
    if (endpoints.empty())
        throw std::invalid_argument("Endpoint list must be non-empty");
    return endpoints[0].protocol();
}

udp_stream::udp_stream(
    io_service_ref io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    const boost::asio::ip::address &interface_address)
    : udp_stream(io_service,
                 make_socket(*io_service, get_protocol(endpoints), interface_address),
                 endpoints, config, buffer_size)
{
}

static boost::asio::ip::udp::socket make_multicast_socket(
    boost::asio::io_service &io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    int ttl)
{
    for (const auto &endpoint : endpoints)
        if (!endpoint.address().is_multicast())
            throw std::invalid_argument("endpoint is not a multicast address");
    boost::asio::ip::udp::socket socket(io_service, get_protocol(endpoints));
    socket.set_option(boost::asio::ip::multicast::hops(ttl));
    return socket;
}

static boost::asio::ip::udp::socket make_multicast_v4_socket(
    boost::asio::io_service &io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    int ttl,
    const boost::asio::ip::address &interface_address)
{
    for (const auto &endpoint : endpoints)
        if (!endpoint.address().is_v4() || !endpoint.address().is_multicast())
            throw std::invalid_argument("endpoint is not an IPv4 multicast address");
    if (!interface_address.is_unspecified() && !interface_address.is_v4())
        throw std::invalid_argument("interface address is not an IPv4 address");
    boost::asio::ip::udp::socket socket(io_service, boost::asio::ip::udp::v4());
    socket.set_option(boost::asio::ip::multicast::hops(ttl));
    if (!interface_address.is_unspecified())
        socket.set_option(boost::asio::ip::multicast::outbound_interface(interface_address.to_v4()));
    return socket;
}

static boost::asio::ip::udp::socket make_multicast_v6_socket(
    boost::asio::io_service &io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    int ttl, unsigned int interface_index)
{
    for (const auto &endpoint : endpoints)
        if (!endpoint.address().is_v6() || !endpoint.address().is_multicast())
            throw std::invalid_argument("endpoint is not an IPv4 multicast address");
    boost::asio::ip::udp::socket socket(io_service, boost::asio::ip::udp::v6());
    socket.set_option(boost::asio::ip::multicast::hops(ttl));
    socket.set_option(boost::asio::ip::multicast::outbound_interface(interface_index));
    return socket;
}

udp_stream::udp_stream(
    io_service_ref io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    int ttl)
    : udp_stream(io_service,
                 make_multicast_socket(*io_service, endpoints, ttl),
                 std::move(endpoints), config, buffer_size)
{
}

udp_stream::udp_stream(
    io_service_ref io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    int ttl,
    const boost::asio::ip::address &interface_address)
    : udp_stream(io_service,
                 make_multicast_v4_socket(*io_service, endpoints, ttl, interface_address),
                 std::move(endpoints), config, buffer_size)
{
}

udp_stream::udp_stream(
    io_service_ref io_service,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size,
    int ttl,
    unsigned int interface_index)
    : udp_stream(io_service,
                 make_multicast_v6_socket(*io_service, endpoints, ttl, interface_index),
                 std::move(endpoints), config, buffer_size)
{
}

udp_stream::udp_stream(
    io_service_ref io_service,
    boost::asio::ip::udp::socket &&socket,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config,
    std::size_t buffer_size)
    : stream(std::make_unique<udp_writer>(
        std::move(io_service),
        std::move(socket),
        endpoints,
        config,
        buffer_size))
{
}

udp_stream::udp_stream(
    io_service_ref io_service,
    boost::asio::ip::udp::socket &&socket,
    const std::vector<boost::asio::ip::udp::endpoint> &endpoints,
    const stream_config &config)
    : udp_stream(io_service, std::move(socket), endpoints, config, 0)
{
}

} // namespace spead2::send
