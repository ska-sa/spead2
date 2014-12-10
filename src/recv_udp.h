#ifndef SPEAD_RECV_UDP_H
#define SPEAD_RECV_UDP_H

#include <cstdint>
#include <boost/asio.hpp>
#include "recv_reader.h"
#include "recv_stream.h"

namespace spead
{
namespace recv
{

class udp_reader : public reader
{
private:
    boost::asio::ip::udp::socket socket;
    // Not used, but need memory available for asio to write to
    boost::asio::ip::udp::endpoint endpoint;
    // Buffer for reading a packet.
    std::unique_ptr<uint8_t[]> buffer;
    std::size_t max_size;

protected:
    void packet_handler(
        const boost::system::error_code &error,
        std::size_t bytes_transferred);

public:
    static constexpr std::size_t default_max_size = 9200;

    explicit udp_reader(
        stream *s,
        boost::asio::io_service &io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        std::size_t max_size = default_max_size,
        std::size_t buffer_size = 0);

    virtual void start(boost::asio::io_service &io_service) override;
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_UDP_H
