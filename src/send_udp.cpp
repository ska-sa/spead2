/* Copyright 2015, 2019-2020 National Research Foundation (SARAO)
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
#include <boost/asio.hpp>
#include <spead2/send_udp.h>
#include <spead2/send_writer.h>
#include <spead2/common_defines.h>
#include <spead2/common_socket.h>

namespace spead2
{
namespace send
{

namespace
{

class udp_writer : public writer
{
private:
    boost::asio::ip::udp::socket socket;
    std::vector<boost::asio::ip::udp::endpoint> endpoints;

    virtual void wakeup() override final;

    static constexpr int max_batch = 64;
#if SPEAD2_USE_SENDMMSG
    struct mmsghdr msgvec[max_batch];
    std::vector<struct iovec> msg_iov;
    transmit_packet packets[max_batch];

    void send_packets(int first, int last);
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

constexpr int udp_writer::max_batch;

#if SPEAD2_USE_SENDMMSG

void udp_writer::send_packets(int first, int last)
{
    // Try sending
    int sent = sendmmsg(socket.native_handle(), msgvec + first, last - first, MSG_DONTWAIT);
    int heaps = 0;
    if (sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
    {
        stream::queue_item *item = packets[first].item;
        if (!item->result)
            item->result = boost::system::error_code(errno, boost::asio::error::get_system_category());
        heaps += packets[first].last;
        first++;
    }
    else if (sent > 0)
    {
        for (int i = 0; i < sent; i++)
        {
            stream::queue_item *item = packets[first].item;
            item->bytes_sent += packets[first].size;
            heaps += packets[first].last;
            first++;
        }
    }

    if (heaps > 0)
        heaps_completed(heaps);
    if (first < last)
    {
        // We didn't manage to send it all: schedule a new attempt once there is
        // buffer space.
        socket.async_send(
            boost::asio::null_buffers(),
            [this, first, last](const boost::system::error_code &, std::size_t) {
                send_packets(first, last);
            });
    }
    else
    {
        post_wakeup();
    }
}

void udp_writer::wakeup()
{
    packet_result result = get_packet(packets[0]);
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

    // We have at least one packet to send. See if we can get some more.
    int n;
    std::size_t n_iov = packets[0].pkt.buffers.size();
    for (n = 1; n < max_batch; n++)
    {
        result = get_packet(packets[n]);
        if (result != packet_result::SUCCESS)
            break;
        n_iov += packets[n].pkt.buffers.size();
    }

    msg_iov.resize(n_iov);
    std::size_t offset = 0;
    for (int i = 0; i < n; i++)
    {
        auto &hdr = msgvec[i].msg_hdr;
        hdr.msg_iov = &msg_iov[offset];
        hdr.msg_iovlen = packets[i].pkt.buffers.size();
        for (const auto &buffer : packets[i].pkt.buffers)
        {
            msg_iov[offset].iov_base = const_cast<void *>(
                boost::asio::buffer_cast<const void *>(buffer));
            msg_iov[offset].iov_len = boost::asio::buffer_size(buffer);
            offset++;
        }
        const auto &endpoint = endpoints[packets[i].item->substream_index];
        hdr.msg_name = (void *) endpoint.data();
        hdr.msg_namelen = endpoint.size();
    }

    send_packets(0, n);
}

#else // SPEAD2_USE_SENDMMSG

void udp_writer::wakeup()
{
    for (int i = 0; i < max_batch; i++)
    {
        transmit_packet data;
        packet_result result = get_packet(data);
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
        stream::queue_item *item = data.item;
        bool last = data.last;
        const auto &endpoint = endpoints[item->substream_index];
        boost::system::error_code ec;
        std::size_t bytes = socket.send_to(data.pkt.buffers, endpoint, 0, ec);
        if (ec == boost::asio::error::would_block)
        {
            // Socket buffer is full, so do an asynchronous send
            auto handler = [this, item, last](const boost::system::error_code &ec, std::size_t bytes_transferred)
            {
                item->bytes_sent += bytes_transferred;
                if (!item->result)
                    item->result = ec;
                if (last)
                    heaps_completed(1);
                wakeup();
            };
            socket.async_send_to(data.pkt.buffers, endpoints[data.item->substream_index],
                                 std::move(handler));
            return;
        }
        else
        {
            item->bytes_sent += bytes;
            if (!item->result)
                item->result = ec;
            if (last)
                heaps_completed(1);
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

constexpr std::size_t udp_stream::default_buffer_size;

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
    : stream(std::unique_ptr<writer>(new udp_writer(
        std::move(io_service),
        std::move(socket),
        endpoints,
        config,
        buffer_size)))
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

} // namespace send
} // namespace spead2
