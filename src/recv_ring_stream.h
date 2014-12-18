/**
 * @file
 */

#ifndef SPEAD_RECV_RING_STREAM
#define SPEAD_RECV_RING_STREAM

#include "common_ringbuffer.h"
#include "common_logging.h"
#include "recv_heap.h"
#include "recv_frozen_heap.h"
#include "recv_stream.h"

namespace spead
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
template<typename Ringbuffer = ringbuffer_cond<heap> >
class ring_stream : public stream
{
private:
    Ringbuffer ready_heaps;

    virtual void heap_ready(heap &&) override;
public:
    /**
     * Constructor. Note that there are two buffers, both of whose size is
     * controlled by @a max_heaps:
     * - the buffer for partial heaps which may still have incoming packets
     * - the buffer for frozen heaps that the consumer has not get dequeued
     * The latter needs to be at least as large as the former to prevent
     * heaps being dropped when the stream is shut down.
     *
     * @param max_heaps The capacity of the ring buffer, and of the stream buffer
     */
    explicit ring_stream(bug_compat_mask bug_compat = 0, std::size_t max_heaps = 16);

    /**
     * Wait until a contiguous heap is available, freeze it, and
     * return it; or until the stream is stopped.
     *
     * @throw ringbuffer_stopped if @ref stop has been called and
     * there are no more contiguous heaps.
     */
    frozen_heap pop();

    /**
     * Like @ref pop, but if no contiguous heap is available,
     * throws @ref ringbuffer_empty.
     *
     * @throw ringbuffer_empty if there is contiguous heap available, but the
     * stream has not been stopped
     * @throw ringbuffer_stopped if @ref stop has been called and
     * there are no more contiguous heaps.
     */
    frozen_heap try_pop();

    virtual void stop() override;

    const Ringbuffer &get_ringbuffer() const { return ready_heaps; }
};

template<typename Ringbuffer>
ring_stream<Ringbuffer>::ring_stream(bug_compat_mask bug_compat, std::size_t max_heaps)
    : stream(bug_compat, max_heaps), ready_heaps(max_heaps)
{
}

template<typename Ringbuffer>
void ring_stream<Ringbuffer>::heap_ready(heap &&h)
{
    try
    {
        ready_heaps.try_push(std::move(h));
    }
    catch (ringbuffer_full &e)
    {
        // Suppress the error, drop the heap
        log_info("dropped heap %d due to insufficient ringbuffer space",
                 h.cnt());
    }
}

template<typename Ringbuffer>
frozen_heap ring_stream<Ringbuffer>::pop()
{
    while (true)
    {
        heap h = ready_heaps.pop();
        if (h.is_contiguous())
            return frozen_heap(std::move(h));
        else
            log_info("received incomplete heap %d", h.cnt());
    }
}

template<typename Ringbuffer>
frozen_heap ring_stream<Ringbuffer>::try_pop()
{
    while (true)
    {
        heap h = ready_heaps.try_pop();
        if (h.is_contiguous())
            return frozen_heap(std::move(h));
        else
            log_info("received incomplete heap %d", h.cnt());
    }
}

template<typename Ringbuffer>
void ring_stream<Ringbuffer>::stop()
{
    /* Note: the order here is important: stream::stop flushes the stream's
     * internal buffer to the ringbuffer, and this needs to happen before
     * the ringbuffer is stopped.
     */
    stream::stop();
    ready_heaps.stop();
}

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_RING_STREAM
