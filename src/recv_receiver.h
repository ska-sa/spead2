#ifndef SPEAD_RECV_RECEIVER_H
#define SPEAD_RECV_RECEIVER_H

#include <type_traits>
#include <future>
#include <vector>
#include <array>
#include <cstdint>
#include <memory>
#include <boost/asio.hpp>
#include "recv_reader.h"

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
    std::vector<std::unique_ptr<reader> > readers;
    std::future<void> worker;

public:
    void add_reader(std::unique_ptr<reader> &&r);

    template<typename T, typename... Args>
    void emplace_reader(Args&&... args)
    {
        std::unique_ptr<reader> ptr(new T(std::forward<Args>(args)...));
        add_reader(std::move(ptr));
    }

    // Run synchronously in the current thread (do not combine with run/stop/join)
    void operator()() { io_service.run(); }
    boost::asio::io_service &get_io_service() { return io_service; }

    void start();  // start running in a separate thread (or threads)
    /* Terminates all streams, blocks until worker threads terminated.
     * Also re-raises any exceptions raised by the thread.
     */
    void stop();
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_RECEIVER_H
