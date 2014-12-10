#ifndef SPEAD_RECV_RECEIVER_H
#define SPEAD_RECV_RECEIVER_H

#include <vector>
#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <boost/asio.hpp>
#include "recv.h"

namespace spead
{
namespace recv
{

/**
 * Single-threaded reception of one or more SPEAD streams.
 */
class receiver
{
private:
    boost::asio::io_service io_service;
    std::vector<std::unique_ptr<stream> > streams;

public:
    void add_stream(std::unique_ptr<stream> &&stream);
    void operator()() { io_service.run(); }
    boost::asio::io_service &get_io_service() { return io_service; }
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_RECEIVER_H
