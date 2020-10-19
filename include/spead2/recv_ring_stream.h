/* Copyright 2015, 2019-2020 National Research Foundation (SARAO)
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

#include <spead2/common_ringbuffer.h>
#include <spead2/common_logging.h>
#include <spead2/common_thread_pool.h>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_heap.h>
#include <spead2/recv_stream.h>
#include <utility>

namespace spead2
{
namespace recv
{

/// Parameters for configuring @ref ring_stream.
class ring_stream_config
{
public:
    static constexpr std::size_t default_heaps = 4;

private:
    std::size_t heaps = default_heaps;
    bool contiguous_only = true;

public:
    /// Set capacity of the ring buffer
    ring_stream_config &set_heaps(std::size_t heaps);
    /// Get capacity of the ring buffer
    std::size_t get_heaps() const { return heaps; }

    /// Set whether only contiguous heaps are pushed to the ring buffer
    ring_stream_config &set_contiguous_only(bool contiguous_only);
    /// Get whether only contiguous heaps are pushed to the ring buffer
    bool get_contiguous_only() const { return contiguous_only; }
};

/**
 * Base class for ring_stream containing only the parts that are independent of
 * the ringbuffer class.
 */
class ring_stream_base : public stream
{
private:
    const ring_stream_config ring_config;

public:
    ring_stream_base(
        io_service_ref io_service,
        const stream_config &config = stream_config(),
        const ring_stream_config &ring_config = ring_stream_config());

    /// Get the ringbuffer configuration passed to the constructor
    const ring_stream_config &get_ring_config() const { return ring_config; }
};

/**
 * Specialisation of @ref stream that pushes its results into a ringbuffer.
 * The ringbuffer class may be replaced, but must provide the same interface as
 * @ref ringbuffer. If the ring buffer fills up, @ref add_packet will block the
 * reader.
 *
 * On the consumer side, heaps are automatically frozen as they are
 * extracted.
 *
 * This class is thread-safe.
 */
template<typename Ringbuffer = ringbuffer<live_heap> >
class ring_stream : public ring_stream_base
{
private:
    Ringbuffer ready_heaps;

    virtual void heap_ready(live_heap &&) override;

public:
    /**
     * Constructor.
     *
     * @param io_service       I/O service (also used by the readers).
     * @param config           Stream configuration
     * @param ring_config      Ringbuffer configuration
     */
    explicit ring_stream(
        io_service_ref io_service,
        const stream_config &config = stream_config(),
        const ring_stream_config &ring_config = ring_stream_config());

    virtual ~ring_stream() override;

    /**
     * Wait until a contiguous heap is available, freeze it, and
     * return it; or until the stream is stopped.
     *
     * @throw ringbuffer_stopped if @ref stop has been called and
     * there are no more contiguous heaps.
     */
    heap pop();

    /**
     * Wait until a heap is available and return it; or until the stream is
     * stopped.
     *
     * @throw ringbuffer_stopped if @ref stop has been called and
     * there are no more heaps.
     */
    live_heap pop_live();

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

    /**
     * Like @ref pop_live, but if no heap is available,
     * throws @ref spead2::ringbuffer_empty.
     *
     * @throw ringbuffer_empty if there is no heap available, but the
     * stream has not been stopped
     * @throw ringbuffer_stopped if @ref stop has been called and
     * there are no more heaps.
     */
    live_heap try_pop_live();

    virtual void stop_received() override;

    virtual void stop() override;

    const Ringbuffer &get_ringbuffer() const { return ready_heaps; }
};

template<typename Ringbuffer>
ring_stream<Ringbuffer>::ring_stream(
    io_service_ref io_service,
    const stream_config &config,
    const ring_stream_config &ring_config)
    : ring_stream_base(std::move(io_service), config, ring_config),
    ready_heaps(ring_config.get_heaps())
{
}

template<typename Ringbuffer>
ring_stream<Ringbuffer>::~ring_stream()
{
    /* See the comments in stop() for why this is necessary. Note that even
     * though stream's destructor calls stop() and stop is virtual, the
     * nature of destructors means that stream's version of stop is called
     * there.
     */
    ready_heaps.stop();
}

template<typename Ringbuffer>
void ring_stream<Ringbuffer>::heap_ready(live_heap &&h)
{
    if (!get_ring_config().get_contiguous_only() || h.is_contiguous())
    {
        try
        {
            try
            {
                ready_heaps.try_push(std::move(h));
            }
            catch (ringbuffer_full &e)
            {
                bool lossy = is_lossy();
                if (lossy)
                    log_warning("worker thread blocked by full ringbuffer on heap %d",
                                h.get_cnt());
                {
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    stats.worker_blocked++;
                }
                ready_heaps.push(std::move(h));
                if (lossy)
                    log_debug("worker thread unblocked, heap %d pushed", h.get_cnt());
            }

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
live_heap ring_stream<Ringbuffer>::pop_live()
{
    return ready_heaps.pop();
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
live_heap ring_stream<Ringbuffer>::try_pop_live()
{
    return ready_heaps.try_pop();
}

template<typename Ringbuffer>
void ring_stream<Ringbuffer>::stop_received()
{
    /* Note: the order here is important: stream::stop_received flushes the
     * stream's internal buffer to the ringbuffer before the ringbuffer is
     * stopped.
     *
     * This only applies to a stop received from the network. A stop received
     * by calling stop() will first stop the ringbuffer to prevent a
     * deadlock.
     */
    stream::stop_received();
    ready_heaps.stop();
}

template<typename Ringbuffer>
void ring_stream<Ringbuffer>::stop()
{
    /* Make sure the ringbuffer is stopped *before* the base implementation
     * takes the queue_mutex. Without this, a heap_ready call could be holding
     * the mutex, waiting for space in the ring buffer. This will cause the
     * heap_ready call to abort, allowing the mutex to be taken for the
     * rest of the shutdown.
     */
    ready_heaps.stop();
    stream::stop();
}

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_RING_STREAM
