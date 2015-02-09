/**
 * @file
 */

#ifndef SPEAD_RECV_READER_H
#define SPEAD_RECV_READER_H

#include <boost/asio.hpp>

namespace spead
{
namespace recv
{

class stream;

/**
 * Abstract base class for asynchronously reading data and passing it into
 * a stream. Subclasses will implement @ref start, and usually @ref stop.
 */
class reader
{
private:
    stream &owner;  ///< Owning stream

public:
    reader(stream &owner)
        : owner(owner) {}
    virtual ~reader() = default;

    /// Retrieve the wrapped stream
    stream &get_stream() const { return owner; }

    /// Retrieve the io_service corresponding to the owner
    boost::asio::io_service &get_io_service();

    /**
     * Enqueue asynchronous operations to the io_service.
     * This function must not send anything to the underlying stream directly;
     * rather, it must arrange for this to happen through the io_service
     * (see @ref get_io_service).
     */
    virtual void start() = 0;

    /**
     * Cancel any pending asynchronous operations. This is called with the
     * owner's strand held.
     */
    virtual void stop() = 0;
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_READER_H
