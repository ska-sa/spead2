#ifndef SPEAD_RECV_MEM_H
#define SPEAD_RECV_MEM_H

#include <cstdint>
#include "recv_reader.h"

namespace spead
{
namespace recv
{

class reader;

class mem_reader : public reader
{
private:
    // Buffer for reading a packet.
    const std::uint8_t *ptr;
    std::size_t length;

public:
    mem_reader(stream *s, const std::uint8_t *ptr, std::size_t length);
    virtual void start(boost::asio::io_service &io_service) override;
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_MEM_H
