#ifndef SPEAD_UDP_IN_H
#define SPEAD_UDP_IN_H

#include <cstdint>
#include <boost/asio.hpp>
#include "in.h"
#include "receiver.h"

namespace spead
{
namespace in
{

class udp_stream : public stream
{
private:
    friend class receiver<udp_stream>;

    receiver<udp_stream> &rec;
    boost::asio::ip::udp::socket socket;
    // Not used, but need memory available for asio to write to
    boost::asio::ip::udp::endpoint endpoint;

protected:
    void ready_handler(
        const boost::system::error_code &error,
        std::size_t bytes_transferred);

    void packet_handler(
        const boost::system::error_code &error,
        std::size_t bytes_transferred);

    void start();

public:
    udp_stream(
        receiver<udp_stream> &rec,
        const boost::asio::ip::udp::endpoint &endpoint);
};

} // namespace in
} // namespace spead

#endif // SPEAD_UDP_IN_H
