#include <cstring>
#include <utility>
#include <cassert>
#include <algorithm>
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

    inline std::int64_t get_id(std::uint64_t pointer) const
    {
        return (pointer >> heap_address_bits) & value_mask;
    }

    inline std::int64_t get_address(std::uint64_t pointer) const
    {
        return pointer & address_mask;
    }

    inline std::int64_t get_value(std::uint64_t pointer) const
    {
        return get_address(pointer);
    }

    inline bool is_immediate(std::uint64_t pointer) const
    {
        return pointer >> 63;
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

// Loads up to 8 bytes as a big-endian number, converts to host endian
static inline std::uint64_t load_bytes_be(const std::uint8_t *ptr, int len)
{
    assert(len <= 8);
    std::uint64_t out = 0;
    std::memcpy(reinterpret_cast<char *>(&out) + 8 - len, ptr, len);
    return be64toh(out);
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
    for (int i = 1; i <= out.n_items; i++)
    {
        uint64_t pointer = be64toh(data64[i]);
        switch (decoder.get_id(pointer))
        {
        case HEAP_CNT_ID:
            out.heap_cnt = decoder.get_value(pointer);
            break;
        case HEAP_LENGTH_ID:
            out.heap_length = decoder.get_value(pointer);
            break;
        case PAYLOAD_OFFSET_ID:
            out.payload_offset = decoder.get_value(pointer);
            break;
        case PAYLOAD_LENGTH_ID:
            out.payload_length = decoder.get_value(pointer);
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
    if (size > payload_reserved)
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
    if (packet.heap_length >= 0 && packet.heap_length < min_length)
        return false;     // inconsistent with already-seen payloads
    if (heap_address_bits != -1 && packet.heap_address_bits != heap_address_bits)
        return false;     // number of heap address bits has changed

    // Packet seems sane, check if we've already seen it, and if not, insert it
    bool new_offset = packet_offsets.insert(packet.payload_offset).second;
    if (!new_offset)
        return false;

    ///////////////////////////////////////////////
    // Packet is now accepted, and we modify state
    ///////////////////////////////////////////////

    heap_address_bits = packet.heap_address_bits;
    // If this is the first time we know the length, record it
    if (heap_length < 0 && packet.heap_length >= 0)
    {
        heap_length = packet.heap_length;
        min_length = heap_length;
        payload_reserve(heap_length, true);
    }
    min_length = std::max(min_length, packet.payload_offset + packet.payload_length);
    pointer_decoder decoder(heap_address_bits);
    for (int i = 0; i < packet.n_items; i++)
    {
        // TODO: should descriptors be put somewhere special to be handled first?
        // TODO: should stream control be handled here?
        std::uint64_t pointer = be64toh(packet.pointers[i]);
        if (decoder.get_id(pointer) > PAYLOAD_LENGTH_ID)
        {
            if (!decoder.is_immediate(pointer))
                min_length = std::max(min_length, std::int64_t(decoder.get_address(pointer)));
            pointers.push_back(pointer);
        }
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

bool heap::is_contiguous() const
{
    return received_length == min_length;
}

///////////////////////////////////////////////////////////////////////

frozen_heap::frozen_heap(heap &&h)
{
    assert(h.is_contiguous());
    /* The length of addressed items is measured from the item to the
     * address of the next item, or the end of the heap. We may receive
     * packets (and hence pointers) out-of-order, so we have to sort.
     * The mask preserves both the address and the immediate-mode flag,
     * so that all addressed items sort together.
     *
     * The sort needs to be stable, because there can be zero-length fields.
     * TODO: these can still break if they cross packet boundaries.
     */
    std::uint64_t sort_mask =
        (std::uint64_t(1) << 63)
        | ((std::uint64_t(1) << h.heap_address_bits) - 1);
    auto compare = [sort_mask](std::uint64_t a, std::uint64_t b) {
        return (a & sort_mask) < (b & sort_mask);
    };
    std::stable_sort(h.pointers.begin(), h.pointers.end(), compare);

    pointer_decoder decoder(h.heap_address_bits);
    items.reserve(h.pointers.size());
    for (std::size_t i = 0; i < h.pointers.size(); i++)
    {
        item new_item;
        std::uint64_t pointer = h.pointers[i];
        new_item.id = decoder.get_id(pointer);
        new_item.is_immediate = decoder.is_immediate(pointer);
        if (new_item.is_immediate)
        {
            new_item.value.immediate = decoder.get_value(pointer);
        }
        else
        {
            std::int64_t start = decoder.get_address(pointer);
            std::int64_t end;
            if (i + 1 < h.pointers.size()
                && !decoder.is_immediate(h.pointers[i + 1]))
                end = decoder.get_address(h.pointers[i + 1]);
            else
                end = h.min_length;
            new_item.value.address.ptr = h.payload.get() + start;
            new_item.value.address.length = end - start;
        }
        items.push_back(new_item);
    }
    heap_cnt = h.heap_cnt;
    heap_address_bits = h.heap_address_bits;
    payload = std::move(h.payload);
    h = heap(0);
}

descriptor frozen_heap::to_descriptor() const
{
    // TODO: unspecified how immediate values are used to encode variable-length fields
    descriptor out;
    for (const item &item : items)
    {
        if (item.is_immediate)
        {
            switch (item.id)
            {
            case DESCRIPTOR_ID_ID:
                out.id = item.value.immediate;
                break;
            default:
                break;
            }
        }
        else
        {
            const std::uint8_t *ptr = item.value.address.ptr;
            std::size_t length = item.value.address.length;
            switch (item.id)
            {
            case DESCRIPTOR_NAME_ID:
                out.name = std::string(reinterpret_cast<const char *>(ptr), length);
                break;
            case DESCRIPTOR_DESCRIPTION_ID:
                out.description = std::string(reinterpret_cast<const char *>(ptr), length);
                break;
            case DESCRIPTOR_FORMAT_ID:
                {
                    int field_size = 9 - heap_address_bits / 8;
                    for (std::size_t i = 0; i + field_size <= length; i++)
                    {
                        char type = ptr[i];
                        std::int64_t bits = load_bytes_be(ptr + i + 1, field_size - 1);
                        out.format.emplace_back(type, bits);
                    }
                    break;
                }
            case DESCRIPTOR_SHAPE_ID:
                {
                    int field_size = 1 + heap_address_bits / 8;
                    for (std::size_t i = 0; i + field_size <= length; i += field_size)
                    {
                        // TODO: the spec and PySPEAD don't agree on how this works
                        bool variable = (ptr[i] & 1);
                        std::int64_t size = load_bytes_be(ptr + i + 1, field_size - 1);
                        out.shape.emplace_back(variable, size);
                    }
                    break;
                }
            case DESCRIPTOR_DTYPE_ID:
                out.dtype = std::string(reinterpret_cast<const char *>(ptr), length);
                break;
            default:
                break;
            }
        }
    }
    // DTYPE overrides format and type
    if (!out.dtype.empty())
    {
        out.shape.clear();
        out.format.clear();
    }
    return out;
}

std::vector<descriptor> frozen_heap::get_descriptors() const
{
    std::vector<descriptor> descriptors;
    auto callback = [&](heap &&h)
    {
        if (h.is_contiguous())
        {
            frozen_heap frozen(std::move(h));
            descriptor d = frozen.to_descriptor();
            if (d.id != 0) // check that we got an ID field
                descriptors.push_back(std::move(d));
        }
    };
    stream descriptor_stream(1);
    descriptor_stream.set_callback(callback);
    for (const item &item : items)
    {
        if (item.id == DESCRIPTOR_ID && !item.is_immediate)
        {
            descriptor_stream.add_packet(item.value.address.ptr, item.value.address.length);
            descriptor_stream.flush();
        }
    }
    return descriptors;
}

void frozen_heap::update_descriptors(descriptor_map &descriptors) const
{
    auto my_descriptors = get_descriptors();
    for (descriptor &d : my_descriptors)
    {
        descriptors[d.id] = std::move(d);
    }
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

void stream::flush()
{
    for (heap &h : heaps)
    {
        callback(std::move(h));
    }
    heaps.clear();
}

} // namespace in
} // namespace spead
