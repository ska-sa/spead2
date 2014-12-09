#ifndef SPEAD_RECEIVER_H
#define SPEAD_RECEIVER_H

#include <vector>
#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <boost/asio.hpp>
#include "defines.h"
#include "in.h"

namespace spead
{
namespace in
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

} // namespace in
} // namespace spead

#endif // SPEAD_RECEIVER_H
