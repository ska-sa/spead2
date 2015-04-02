/**
 * @file
 */

#ifndef SPEAD2_SEND_UDP_H
#define SPEAD2_SEND_UDP_H

#include <boost/asio.hpp>
#include <utility>
#include "send_packet.h"
#include "send_stream.h"

namespace spead2
{
namespace send
{

class udp_stream : public stream<udp_stream>
{
private:
    friend class stream<udp_stream>;
    boost::asio::ip::udp::socket socket;
    boost::asio::ip::udp::endpoint endpoint;

    template<typename Handler>
    void async_send_packet(const packet &pkt, Handler &&handler)
    {
        socket.async_send_to(pkt.buffers, endpoint, std::move(handler));
    }

public:
    /// Socket receive buffer size, if none is explicitly passed to the constructor
    static constexpr std::size_t default_buffer_size = 512 * 1024;

    /// Constructor
    udp_stream(
        boost::asio::io_service &io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config = stream_config(),
        std::size_t buffer_size = default_buffer_size);
};

} // namespace send
} // namespace spead2

#endif // SPEAD2_SEND_UDP_H
