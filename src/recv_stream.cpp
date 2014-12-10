#include <cstddef>
#include <utility>
#include "recv_stream.h"
#include "recv_heap.h"

namespace spead
{
namespace recv
{

stream::stream(std::size_t max_heaps) : max_heaps(max_heaps)
{
}

void stream::set_max_heaps(std::size_t max_heaps)
{
    this->max_heaps = max_heaps;
}

bool stream::add_packet(const packet_header &packet)
{
    // Look for matching heap
    auto insert_before = heaps.begin();
    for (auto it = heaps.begin(); it != heaps.end(); ++it)
    {
        heap &h = *it;
        if (h.cnt() == packet.heap_cnt)
        {
            bool result = h.add_packet(packet);
            if (result && h.is_complete())
            {
                heap_ready(std::move(h));
                heaps.erase(it);
            }
            return result;
        }
        else if (h.cnt() < packet.heap_cnt)
            insert_before = next(it);
    }

    // Doesn't match any previously seen heap, so create a new one
    heap h(packet.heap_cnt);
    if (!h.add_packet(packet))
        return false; // probably unreachable, since decode_packet already validates
    if (h.is_complete())
    {
        heap_ready(std::move(h));
    }
    else
    {
        heaps.insert(insert_before, std::move(h));
        if (heaps.size() > max_heaps)
        {
            // Too many active heaps: pop the lowest ID, even if incomplete
            heap_ready(std::move(heaps[0]));
            heaps.pop_front();
        }
    }
    return true;
}

void stream::flush()
{
    for (heap &h : heaps)
    {
        heap_ready(std::move(h));
    }
    heaps.clear();
}

void stream::end_of_stream()
{
    flush();
    heap_ready(heap(0)); // mark end of stream
}

const void *mem_to_stream(stream &s, const void *ptr, std::size_t length)
{
    const std::uint8_t *p = reinterpret_cast<const std::uint8_t *>(ptr);
    while (length > 0)
    {
        packet_header packet;
        std::size_t size = decode_packet(packet, p, length);
        if (size > 0)
        {
            s.add_packet(packet);
            p += size;
            length -= size;
        }
        else
            length = 0; // causes loop to exit
    }
    return p;
}

} // namespace recv
} // namespace spead
