/* Copyright 2015 SKA South Africa
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 */

#ifndef SPEAD2_RECV_READER_H
#define SPEAD2_RECV_READER_H

#include <boost/asio.hpp>
#include <future>
#include <utility>

namespace spead2
{
namespace recv
{

class stream;
class stream_base;

/**
 * Abstract base class for asynchronously reading data and passing it into
 * a stream. Subclasses will usually override @ref stop.
 *
 * The lifecycle of a reader is:
 * - construction (strand held)
 * - stop (strand held)
 * - join (strand not held)
 * - destruction (strand held)
 */
class reader
{
private:
    stream &owner;  ///< Owning stream
    std::promise<void> stopped_promise; ///< Promise filled when last completion handler done

protected:
    /// Called by last completion handler
    void stopped();

public:
    explicit reader(stream &owner) : owner(owner) {}
    virtual ~reader() = default;

    /// Retrieve the wrapped stream
    stream &get_stream() const { return owner; }

    /**
     * Retrieve the wrapped stream's base class. This must only be used when
     * the stream's strand is held.
     */
    stream_base &get_stream_base() const;

    /// Retrieve the io_service corresponding to the owner
    boost::asio::io_service &get_io_service();

    /**
     * Cancel any pending asynchronous operations. This is called with the
     * owner's strand held. This function does not need to wait for
     * completion handlers to run, but if there are any, the destructor must
     * wait for them.
     */
    virtual void stop() = 0;

    /**
     * Block until @ref stopped has been called by the last completion
     * handler. This function is called outside the strand.
     */
    void join();
};

/**
 * Factory for creating a new reader. This is used by @ref
 * stream::emplace_reader to create the reader. The default implementation
 * simply chains to the constructor, but it can be overloaded in cases where
 * it is desirable to select the class dynamically.
 */
template<typename Reader>
struct reader_factory
{
    template<typename... Args>
    static std::unique_ptr<reader> make_reader(Args&&... args)
    {
        return std::unique_ptr<reader>(new Reader(std::forward<Args>(args)...));
    }
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_READER_H
