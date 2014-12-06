#include <cstring>
#include <utility>
#include <cassert>
#include <endian.h>
#include "in.h"

namespace spead
{
namespace in
{

namespace
{

class pointer_decoder
{
private:
    int heap_address_bits;
    std::uint64_t address_mask;
    std::uint64_t value_mask;

public:
    explicit pointer_decoder(int heap_address_bits)
    {
        this->heap_address_bits = heap_address_bits;
        this->address_mask = (std::uint64_t(1) << heap_address_bits) - 1;
        this->value_mask = (std::uint64_t(1) << (63 - heap_address_bits)) - 1;
    }

    inline std::uint64_t get_id(std::uint64_t descr) const
    {
        return (descr >> heap_address_bits) & value_mask;
    }

    inline std::uint64_t get_address(std::uint64_t descr) const
    {
        return descr & address_mask;
    }

    inline std::uint64_t get_value(std::uint64_t descr) const
    {
        return get_address(descr);
    }

    inline bool is_immediate(std::uint64_t descr) const
    {
        return descr >> 63;
    }

    inline int address_bits() const
    {
        return heap_address_bits;
    }
};

// cnt must be strictly less than 64
static inline std::uint64_t extract_bits(std::uint64_t value, int first, int cnt)
{
    return (value >> first) & ((std::uint64_t(1) << cnt) - 1);
}

} // anonymous namespace

///////////////////////////////////////////////////////////////////////

std::size_t decode_packet(packet_header &out, const uint8_t *data, std::size_t max_size)
{
    if (max_size < 8)
        return 0; // too small
    const uint64_t *data64 = reinterpret_cast<const uint64_t *>(data);
    uint64_t header = be64toh(data64[0]);
    if (extract_bits(header, 48, 16) != magic_version)
        return 0;
    int item_id_bits = extract_bits(header, 40, 8) * 8;
    int heap_address_bits = extract_bits(header, 32, 8) * 8;
    if (item_id_bits == 0 || heap_address_bits == 0)
        return 0;             // not really legal
    if (item_id_bits + heap_address_bits != 64)
        return 0;             // not SPEAD-64-*, which is what we support

    out.n_items = extract_bits(header, 0, 16);
    if (std::size_t(out.n_items) * 8 + 8 > max_size)
        return 0;             // not enough space for all the item pointers

    // Mark specials as not found
    out.heap_cnt = -1;
    out.heap_length = -1;
    out.payload_offset = -1;
    out.payload_length = -1;
    pointer_decoder decoder(heap_address_bits);
    for (unsigned int i = 1; i <= out.n_items; i++)
    {
        uint64_t descr = be64toh(data64[i]);
        switch (decoder.get_id(descr))
        {
        case HEAP_CNT_ID:
            out.heap_cnt = decoder.get_value(descr);
            break;
        case HEAP_LENGTH_ID:
            out.heap_length = decoder.get_value(descr);
            break;
        case PAYLOAD_OFFSET_ID:
            out.payload_offset = decoder.get_value(descr);
            break;
        case PAYLOAD_LENGTH_ID:
            out.payload_length = decoder.get_value(descr);
            break;
        default:
            break;
        }
    }
    // Certain specials are required
    if (out.heap_cnt == -1 || out.payload_offset == -1 || out.payload_length == -1)
        return 0;
    // Packet length must fit
    std::size_t size = out.payload_length + out.n_items * 8 + 8;
    if (size > max_size)
        return 0;
    // If a heap length is given, the payload must fit
    if (out.heap_length >= 0 && out.payload_offset + out.payload_length > out.heap_length)
        return 0;

    out.pointers = data64 + 1;
    out.payload = data + (out.n_items * 8 + 8);
    out.heap_address_bits = heap_address_bits;
    return size;
}

heap::heap(std::int64_t heap_cnt) : heap_cnt(heap_cnt)
{
    assert(heap_cnt >= 0);
}

void heap::payload_reserve(std::size_t size, bool exact)
{
    if (size >= payload_reserved)
    {
        if (!exact && size < payload_reserved * 2)
        {
            size = payload_reserved * 2;
        }
        std::unique_ptr<std::uint8_t[]> new_payload(new std::uint8_t[size]);
        if (payload)
            std::memcpy(new_payload.get(), payload.get(), payload_reserved);
        payload = std::move(new_payload);
        payload_reserved = size;
    }
}

bool heap::add_packet(const packet_header &packet)
{
    if (heap_cnt != packet.heap_cnt)
        return false;
    if (heap_length >= 0
        && packet.heap_length >= 0
        && packet.heap_length != heap_length)
        return false; // inconsistent heap lengths - could cause trouble later
    if (packet.heap_length >= 0 && std::size_t(packet.heap_length) < payload_reserved)
        return false;     // inconsistent with already-seen payloads
    if (heap_address_bits != -1 && packet.heap_address_bits != heap_address_bits)
        return false;     // number of heap address bits has changed

    // Packet seems sane, check if we've already seen it, and if not, insert it
    bool new_offset = packet_offsets.insert(packet.payload_offset).second;
    if (!new_offset)
        return false;

    heap_address_bits = packet.heap_address_bits;
    // If this is the first time we know the length, record it
    if (heap_length < 0 && packet.heap_length >= 0)
    {
        heap_length = packet.heap_length;
        payload_reserve(heap_length, true);
    }
    pointer_decoder decoder(heap_address_bits);
    for (unsigned int i = 0; i < packet.n_items; i++)
    {
        // TODO: should descriptors be put somewhere special to be handled first?
        // TODO: should stream control be handled here?
        if (decoder.get_id(packet.pointers[i]) > PAYLOAD_LENGTH_ID)
            pointers.push_back(be64toh(packet.pointers[i]));
    }

    if (packet.payload_length > 0)
    {
        std::memcpy(payload.get() + packet.payload_offset,
                    packet.payload,
                    packet.payload_length);
        received_length += packet.payload_length;
    }
    return true;
}

bool heap::is_complete() const
{
    return received_length == heap_length;
}

///////////////////////////////////////////////////////////////////////

stream::stream(std::size_t max_heaps) : max_heaps(max_heaps)
{
}

void stream::set_max_heaps(std::size_t max_heaps)
{
    this->max_heaps = max_heaps;
}

void stream::set_callback(std::function<void(heap &&)> callback)
{
    this->callback = std::move(callback);
}

bool stream::add_packet(const uint8_t *data, std::size_t size)
{
    packet_header packet;
    std::size_t real_size = decode_packet(packet, data, size);
    if (real_size == 0 || real_size != size)
        return false; // corrupt packet

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
                if (callback)
                    callback(std::move(h));
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
        callback(std::move(h));
    }
    else
    {
        heaps.insert(insert_before, std::move(h));
        if (heaps.size() > max_heaps)
        {
            // Too many active heaps: pop the lowest ID, even if incomplete
            callback(std::move(heaps[0]));
            heaps.pop_front();
        }
    }
    return true;
}

} // namespace in
} // namespace spead
