#ifndef SPEAD_RECV_MEM_H
#define SPEAD_RECV_MEM_H

#include <cstdint>
#include <boost/asio.hpp>
#include "recv.h"

namespace spead
{
namespace recv
{

class mem_stream : public stream
{
private:
    // Buffer for reading a packet.
    const std::uint8_t *ptr;
    std::size_t length;

public:
    mem_stream(const std::uint8_t *ptr, std::size_t length);
    void run();
    virtual void start() override { run(); }
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_MEM_H
