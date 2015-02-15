#include <boost/asio.hpp>
#include "send_heap.h"
#include "send_rate_writer.h"

namespace spead
{
namespace send
{

class udp_writer : public rate_writer<udp_writer>
{
private:
    friend class rate_writer<udp_writer>;

    boost::asio::ip::udp::socket socket;

    template<typename Handler>
    void async_send_packet(const packet &pkt, Handler &&handler)
    {
        socket.async_send(pkt.buffers, std::move(handler));
    }

public:
    udp_writer(boost::asio::io_service &io_service, const boost::asio::ip::udp::endpoint &endpoint)
        : rate_writer<udp_writer>(io_service), socket(io_service)
    {
        socket.connect(endpoint);
    }
};

} // namespace send
} // namespace spead
