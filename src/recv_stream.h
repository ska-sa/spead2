#ifndef SPEAD_RECV_STREAM
#define SPEAD_RECV_STREAM

#include <cstddef>
#include <deque>
#include "recv_heap.h"

namespace spead
{
namespace recv
{

class packet_header;

class stream
{
private:
    // TODO: replace with a fixed-size ring buffer
    std::size_t max_heaps;
    std::deque<heap> heaps;

private:
    // Called when a heap is ready for further processing
    // End-of-stream is indicated by an empty heap
    virtual void heap_ready(heap &&) {}

public:
    explicit stream(std::size_t max_heaps = 16);
    virtual ~stream() = default;

    void set_max_heaps(std::size_t max_heaps);

    bool add_packet(const packet_header &packet);
    // Mark end-of-stream (implicitly does a flush)
    void end_of_stream();
    // Clear out all heaps from the deque, even if not complete
    void flush();
};

/* Push packets found in a block to memory to a stream.
 * Returns a pointer to after the last valid packet.
 */
const void *mem_to_stream(stream &s, const void *ptr, std::size_t length);

} // namespace recv
} // namespace spead

#endif // SPEAD_RECV_STREAM
