#include <endian.h>
#include <algorithm>
#include <cassert>
#include <utility>
#include <cstring>
#include <string>
#include "recv_heap.h"
#include "recv_frozen_heap.h"
#include "recv_stream.h"
#include "recv_utils.h"

namespace spead
{
namespace recv
{

// Loads up to 8 bytes as a big-endian number, converts to host endian
static inline std::uint64_t load_bytes_be(const std::uint8_t *ptr, int len)
{
    assert(len <= 8);
    std::uint64_t out = 0;
    std::memcpy(reinterpret_cast<char *>(&out) + 8 - len, ptr, len);
    return be64toh(out);
}

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
        if (new_item.id == 0)
            continue; // just padding
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
#if WORKAROUND_SR_96
                    int field_size = 4;
#else
                    int field_size = 9 - heap_address_bits / 8;
#endif
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
#if WORKAROUND_SR_96
                    int field_size = 8;
#else
                    int field_size = 1 + heap_address_bits / 8;
#endif
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

namespace
{

class descriptor_stream : public stream
{
private:
    virtual void heap_ready(heap &&h) override;
public:
    std::vector<descriptor> descriptors;
};

void descriptor_stream::heap_ready(heap &&h)
{
    if (h.is_contiguous())
    {
        frozen_heap frozen(std::move(h));
        descriptor d = frozen.to_descriptor();
        if (d.id != 0) // check that we got an ID field
            descriptors.push_back(std::move(d));
    }
}

} // anonymous namespace

std::vector<descriptor> frozen_heap::get_descriptors() const
{
    descriptor_stream s;
    for (const item &item : items)
    {
        if (item.id == DESCRIPTOR_ID && !item.is_immediate)
        {
            mem_to_stream(s, item.value.address.ptr, item.value.address.length);
            s.stop();
        }
    }
    return s.descriptors;
}

} // namespace recv
} // namespace spead
