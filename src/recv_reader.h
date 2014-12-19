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
    boost::asio::strand strand; ///< Serialises access to this structure
    stream &s;  ///< Wrapped stream

public:
    reader(boost::asio::io_service &io_service, stream &s)
        : strand(io_service), s(s) {}
    virtual ~reader() = default;

    /// Retrieve the wrapped stream
    stream &get_stream() const { return s; }

    /// Retrieve the internal strand
    boost::asio::strand &get_strand() { return strand; }

    /**
     * Enqueue asynchronous operations to the io_service.
     * This function must not send anything to the underlying stream directly;
     * rather, it must arrange for this to happen through the io_service
     * (see @ref get_io_service).
     */
    virtual void start() = 0;

    /**
     * Cancel any pending asynchronous operations. Overrides must chain
     * back to the base class, which takes care of stopping the wrapped
     * stream.
     */
    virtual void stop();
};

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_READER_H
