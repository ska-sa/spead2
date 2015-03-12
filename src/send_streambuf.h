/**
 * @file
 */

#ifndef SPEAD_SEND_STREAMBUF_H
#define SPEAD_SEND_STREAMBUF_H

#include <streambuf>
#include <boost/asio.hpp>
#include "send_stream.h"

namespace spead
{
namespace send
{

class streambuf_stream : public stream<streambuf_stream>
{
private:
    friend class stream<streambuf_stream>;
    std::streambuf &streambuf;

    template<typename Handler>
    void async_send_packet(const packet &pkt, Handler &&handler)
    {
        for (const auto &buffer : pkt.buffers)
            streambuf.sputn(boost::asio::buffer_cast<const char *>(buffer),
                            boost::asio::buffer_size(buffer));
        get_io_service().dispatch(std::move(handler));
    }

public:
    streambuf_stream(
        boost::asio::io_service &io_service,
        std::streambuf &streambuf,
        int heap_address_bits,
        std::size_t max_packet_size,
        double rate = 0.0,
        std::size_t max_heaps = default_max_heaps);
};

} // namespace send
} // namespace spead

#endif // SPEAD_SEND_STREAMBUF_H
