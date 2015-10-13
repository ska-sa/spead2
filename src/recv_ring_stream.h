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

#ifndef SPEAD2_RECV_RING_STREAM
#define SPEAD2_RECV_RING_STREAM

#include "common_ringbuffer.h"
#include "common_logging.h"
#include "common_thread_pool.h"
#include "recv_live_heap.h"
#include "recv_heap.h"
#include "recv_stream.h"

namespace spead2
{
namespace recv
{

/**
 * Specialisation of @ref stream that pushes its results into a ringbuffer.
 * The ringbuffer class may be replaced, but must provide the same interface
 * as @ref ringbuffer_cond. If the ring buffer fills up, new heaps are
 * discarded, rather than blocking the receiver.
 *
 * On the consumer side, heaps are automatically frozen as they are
 * extracted.
 *
 * This class is thread-safe.
 */
template<typename Ringbuffer = ringbuffer<live_heap> >
class ring_stream : public stream
{
private:
    Ringbuffer ready_heaps;
    bool contiguous_only;

    virtual void heap_ready(live_heap &&) override;
public:
    /**
     * Constructor. Note that there are two buffers, both of whose size is
     * controlled by @a max_heaps:
     * - the buffer for partial heaps which may still have incoming packets
     * - the buffer for frozen heaps that the consumer has not get dequeued
     * The latter needs to be at least as large as the former to prevent
     * heaps being dropped when the stream is shut down.
     */
    explicit ring_stream(
        boost::asio::io_service &io_service,
        bug_compat_mask bug_compat = 0,
        std::size_t max_heaps = default_max_heaps,
        bool contiguous_only = true);
    explicit ring_stream(
        thread_pool &pool,
        bug_compat_mask bug_compat = 0,
        std::size_t max_heaps = default_max_heaps,
        bool contiguous_only = true)
        : ring_stream(pool.get_io_service(), bug_compat, max_heaps, contiguous_only) {}

    /**
     * Wait until a contiguous heap is available, freeze it, and
     * return it; or until the stream is stopped.
     *
     * @throw ringbuffer_stopped if @ref stop has been called and
     * there are no more contiguous heaps.
     */
    heap pop();

    /**
     * Like @ref pop, but if no contiguous heap is available,
     * throws @ref spead2::ringbuffer_empty.
     *
     * @throw ringbuffer_empty if there is no contiguous heap available, but the
     * stream has not been stopped
     * @throw ringbuffer_stopped if @ref stop has been called and
     * there are no more contiguous heaps.
     */
    heap try_pop();

    virtual void stop_received() override;

    virtual void stop() override;

    const Ringbuffer &get_ringbuffer() const { return ready_heaps; }
};

template<typename Ringbuffer>
ring_stream<Ringbuffer>::ring_stream(
    boost::asio::io_service &io_service,
    bug_compat_mask bug_compat, std::size_t max_heaps,
    bool contiguous_only)
    : stream(io_service, bug_compat, max_heaps), ready_heaps(2),
    contiguous_only(contiguous_only)
{
}

template<typename Ringbuffer>
void ring_stream<Ringbuffer>::heap_ready(live_heap &&h)
{
    if (!contiguous_only || h.is_contiguous())
    {
        try
        {
            ready_heaps.push(std::move(h));
        }
        catch (ringbuffer_stopped &e)
        {
            // Suppress the error, drop the heap
            log_info("dropped heap %d due to external stop",
                     h.get_cnt());
        }
    }
    else
    {
        log_warning("dropped incomplete heap %d (%d/%d bytes of payload)",
                    h.get_cnt(), h.get_received_length(), h.get_heap_length());
    }
}

template<typename Ringbuffer>
heap ring_stream<Ringbuffer>::pop()
{
    while (true)
    {
        live_heap h = ready_heaps.pop();
        if (h.is_contiguous())
            return heap(std::move(h));
        else
            log_info("received incomplete heap %d", h.get_cnt());
    }
}

template<typename Ringbuffer>
heap ring_stream<Ringbuffer>::try_pop()
{
    while (true)
    {
        live_heap h = ready_heaps.try_pop();
        if (h.is_contiguous())
            return heap(std::move(h));
        else
            log_info("received incomplete heap %d", h.get_cnt());
    }
}

template<typename Ringbuffer>
void ring_stream<Ringbuffer>::stop_received()
{
    ready_heaps.stop();
    stream::stop_received();
}

template<typename Ringbuffer>
void ring_stream<Ringbuffer>::stop()
{
    /* Make sure the ringbuffer is stopped *before* the base implementation
     * takes the strand. Without this, a heap_ready call could be blocking the
     * strand, waiting for space in the ring buffer. This will call the
     * heap_ready call to abort, allowing the strand to be accessed for the
     * rest of the shutdown.
     */
    ready_heaps.stop();
    stream::stop();
}

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_RING_STREAM
