#ifndef SPEAD_RECV_RING_STREAM
#define SPEAD_RECV_RING_STREAM

#include "common_ringbuffer.h"
#include "recv_heap.h"
#include "recv_frozen_heap.h"
#include "recv_stream.h"

namespace spead
{
namespace recv
{

template<typename Ringbuffer = ringbuffer<heap> >
class ring_stream : public stream
{
private:
    Ringbuffer ready_heaps;

    virtual void heap_ready(heap &&) override;
public:
    explicit ring_stream(std::size_t max_heaps = 16);
    frozen_heap pop();
    virtual void stop() override;
};

template<typename Ringbuffer>
ring_stream<Ringbuffer>::ring_stream(std::size_t max_heaps)
    : stream(max_heaps), ready_heaps(max_heaps)
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
        // TODO: log it?
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
    }
}

template<typename Ringbuffer>
void ring_stream<Ringbuffer>::stop()
{
    stream::stop();
    ready_heaps.stop();
}

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_RING_STREAM
