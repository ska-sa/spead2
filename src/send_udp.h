/**
 * @file
 */

#ifndef SPEAD_SEND_UDP_H
#define SPEAD_SEND_UDP_H

#include <boost/asio.hpp>
#include "send_packet.h"
#include "send_stream.h"

namespace spead
{
namespace send
{

class udp_stream : public stream<udp_stream>
{
private:
    friend class stream<udp_stream>;
    boost::asio::ip::udp::socket socket;

    template<typename Handler>
    void async_send_packet(const packet &pkt, Handler &&handler)
    {
        socket.async_send(pkt.buffers, std::move(handler));
    }

public:
    udp_stream(
        boost::asio::ip::udp::socket &&socket,
        int heap_address_bits,
        bug_compat_mask bug_compat,
        std::size_t max_packet_size,
        double rate,
        std::size_t max_heaps = DEFAULT_MAX_HEAPS)
        : stream<udp_stream>(socket.get_io_service(), heap_address_bits,
                             bug_compat, max_packet_size, rate, max_heaps),
        socket(std::move(socket))
    {
    }

    udp_stream(
        boost::asio::io_service &io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        int heap_address_bits,
        bug_compat_mask bug_compat,
        std::size_t max_packet_size,
        double rate,
        std::size_t max_heaps = DEFAULT_MAX_HEAPS)
        : udp_stream(
            boost::asio::ip::udp::socket(io_service),
            heap_address_bits, bug_compat, max_packet_size, rate, max_heaps)
    {
        socket.connect(endpoint);
    }
};

} // namespace send
} // namespace spead

#endif // SPEAD_SEND_UDP_H
