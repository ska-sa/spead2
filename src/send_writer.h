/**
 * @file
 */

#ifndef SPEAD_SEND_WRITER_H
#define SPEAD_SEND_WRITER_H

#include <boost/asio.hpp>
#include <functional>

namespace spead
{
namespace send
{

class heap;

class writer
{
private:
    boost::asio::io_service &io_service;

public:
    explicit writer(boost::asio::io_service &io_service);
    virtual ~writer() = default;

    boost::asio::io_service &get_io_service() const { return io_service; }

    virtual void async_send_heap(const heap &h, write_handler &handler);
};

} // namespace send
} // namespace spead

#endif // SPEAD_SEND_WRITER_H
