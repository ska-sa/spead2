#ifndef SPEAD_RECEIVER_H
#define SPEAD_RECEIVER_H

#include <vector>
#include <array>
#include <cstdint>
#include <boost/asio.hpp>
#include "defines.h"
#include "in.h"

namespace spead
{
namespace in
{

template<typename Stream>
class receiver;

/**
 * Single-threaded reception of one or more SPEAD streams. The Stream
 * can implement any protocol, but it must be movable.
 */
template<typename Stream>
class receiver
{
public:
    typedef Stream stream_type;
    friend Stream;
private:
    boost::asio::io_service io_service;
    std::vector<Stream> streams;
    // This is provided for the stream class to use. It has one extra byte so
    // that truncation can be detected.
    alignas(std::uint64_t) std::array<uint8_t, max_packet_length + 1> buffer;

public:
    void add_stream(Stream &&stream);
    void operator()();
};

template<typename Stream>
void receiver<Stream>::add_stream(Stream &&stream)
{
    streams.push_back(std::move(stream));
    streams.back().start();
}

template<typename Stream>
void receiver<Stream>::operator()()
{
    io_service.run();
}

} // namespace in
} // namespace spead

#endif // SPEAD_RECEIVER_H
