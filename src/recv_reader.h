#ifndef SPEAD_RECV_READER_H
#define SPEAD_RECV_READER_H

#include <boost/asio.hpp>

namespace spead
{
namespace recv
{

class stream;

class reader
{
private:
    stream *s;

public:
    explicit reader(stream *s) : s(s) {}
    virtual ~reader() = default;

    stream *get_stream() const { return s; }

    virtual void start(boost::asio::io_service &io_service) = 0;
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_READER_H
