#ifndef SPEAD_MEM_IN_H
#define SPEAD_MEM_IN_H

#include <cstdint>
#include <boost/asio.hpp>
#include "in.h"

namespace spead
{
namespace in
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

} // namespace in
} // namespace spead

#endif // SPEAD_MEM_IN_H
