/**
 * @file
 */

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
#include "common_logging.h"

namespace spead
{
namespace recv
{

/**
 * Read @a len bytes from @a ptr, and interpret them as a big-endian number.
 * It is not necessary for @a ptr to be aligned.
 *
 * @pre @a 0 &lt;= len &lt;= 8
 */
static inline std::uint64_t load_bytes_be(const std::uint8_t *ptr, int len)
{
    assert(0 <= len && len <= 8);
    std::uint64_t out = 0;
    std::memcpy(reinterpret_cast<char *>(&out) + 8 - len, ptr, len);
    return be64toh(out);
}

frozen_heap::frozen_heap(heap &&h)
{
    assert(h.is_contiguous());
    log_debug("freezing heap with ID %d, %d item pointers, %d bytes payload",
              h.cnt(), h.pointers.size(), h.min_length);
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
            new_item.value.immediate = decoder.get_immediate(pointer);
            log_debug("Found new immediate item ID %d, value %d", new_item.id, new_item.value.immediate);
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
            log_debug("Found new addressed item ID %d, offset %d, length %d",
                    start, end - start);
        }
        items.push_back(new_item);
    }
    heap_cnt = h.heap_cnt;
    heap_address_bits = h.heap_address_bits;
    payload = std::move(h.payload);
    // Reset h so that it still satisfies its invariants
    h = heap(0);
}

descriptor frozen_heap::to_descriptor() const
{
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
                log_info("Unrecognised descriptor item ID %x", item.id);
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
#if BUG_COMPAT_DESCRIPTOR_WIDTHS
                    int field_size = 4;
#else
                    int field_size = 9 - heap_address_bits / 8;
#endif
                    for (std::size_t i = 0; i + field_size <= length; i += field_size)
                    {
                        char type = ptr[i];
                        std::int64_t bits = load_bytes_be(ptr + i + 1, field_size - 1);
                        out.format.emplace_back(type, bits);
                    }
                    break;
                }
            case DESCRIPTOR_SHAPE_ID:
                {
#if BUG_COMPAT_DESCRIPTOR_WIDTHS
                    int field_size = 8;
#else
                    int field_size = 1 + heap_address_bits / 8;
#endif
                    for (std::size_t i = 0; i + field_size <= length; i += field_size)
                    {
#if BUG_COMPAT_SHAPE_BIT_1
                        bool variable = (ptr[i] & 2);
#else
                        bool variable = (ptr[i] & 1);
#endif
                        std::int64_t size = variable ? -1 : load_bytes_be(ptr + i + 1, field_size - 1);
                        out.shape.push_back(size);
                    }
                    break;
                }
            case DESCRIPTOR_DTYPE_ID:
                out.numpy_header = std::string(reinterpret_cast<const char *>(ptr), length);
                break;
            default:
                log_info("Unrecognised descriptor item ID %x", item.id);
                break;
            }
        }

    }
    // DTYPE overrides format and type
    if (!out.numpy_header.empty())
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
        else
            log_info("incomplete descriptor (no ID)");
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
