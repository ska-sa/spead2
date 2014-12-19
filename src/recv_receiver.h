/**
 * @file
 */

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
 * Single-threaded reception of one or more SPEAD streams. This class creates
 * a separate thread that listens for packets and dispatches them to streams.
 * The actual protocol details are encapsulated in subclasses of @ref reader.
 * Alternatively, it can be run in the calling thread.
 *
 * The receiver owns the readers, but not the streams. Not owning the streams
 * simplifies the Python wrapper, where these are separate objects, and
 * lifetime is managed via Python. The readers must be owned by this class
 * because they use the embedded @c io_service.
 *
 * New readers can be added while the receiver is running, but they are never
 * removed. Thus, if the program flow involves continually adding new readers
 * and streams over time, it is better to periodically throw away the
 * receivers. If this becomes a problem, it is technical possible to allow
 * readers to be removed when they have terminated, but it would require
 * additional locks that may harm performance.
 *
 * @warning This class is @em not fully thread-safe. All calls to the public
 * interface must be made serially, except where otherwise noted.
 */
class receiver
{
private:
    boost::asio::io_service io_service;
    /// List of readers managed by this class
    std::vector<std::unique_ptr<reader> > readers;
    /**
     * Futures that becomes ready when the worker (either operator()()
     * or a thread) completes. In the former case it is connected to
     * a promise, in the latter case to an async task.
     */
    std::vector<std::future<void> > workers;

public:
    ~receiver();

    /// Retrieve the embedded io_service
    boost::asio::io_service &get_io_service() { return io_service; }

    /**
     * Add a new reader by passing its constructor arguments, excluding
     * the initial @a io_service argument.
     *
     * @todo Check that @a s is not already managed by another reader.
     */
    template<typename T, typename... Args>
    void emplace_reader(stream &s, Args&&... args)
    {
        reader *r = new T(io_service, s, std::forward<Args>(args)...);
        std::unique_ptr<reader> ptr(r);
        readers.push_back(std::move(ptr));
        r->start();
    }

    /**
     * Run reactor in the current thread. In this case, @ref start must
     * not be used, but @ref stop @em must be called to synchronise state.
     * It may either be called from another thread (to shut down the
     * reactor), or it may be called from this thread after this function
     * returns (which will happen once all readers reach end-of-stream).
     */
    void operator()();

    /**
     * Run the reactor in a separate thread. This call returns immediately.
     *
     * @throw std::invalid_argument if the reactor is already running
     */
    void start(int num_threads = 1);

    /**
     * Shuts down the receiver:
     * - Stops all readers
     * - Waits for the reactor to shut down and any worker thread to exit
     * - Removes all readers
     * - If the receiver was launched by calling @ref start, re-raises any
     *   exception thrown in the worker thread.
     *
     * After this returns, it is possible to add new readers and start again.
     */
    void stop();
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_RECEIVER_H
