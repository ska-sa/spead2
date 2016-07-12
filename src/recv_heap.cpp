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

#include <algorithm>
#include <cassert>
#include <utility>
#include <cstring>
#include <string>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_heap.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_utils.h>
#include <spead2/common_logging.h>
#include <spead2/common_endian.h>

namespace spead2
{
namespace recv
{

/**
 * Read @a len bytes from @a ptr, and interpret them as a big-endian number.
 * It is not necessary for @a ptr to be aligned.
 *
 * @pre @a 0 &lt;= len &lt;= sizeof(item_pointer_t)
 */
static inline item_pointer_t load_bytes_be(const std::uint8_t *ptr, int len)
{
    assert(0 <= len && len <= sizeof(item_pointer_t));
    item_pointer_t out = 0;
    std::memcpy(reinterpret_cast<char *>(&out) + sizeof(item_pointer_t) - len, ptr, len);
    return betoh<item_pointer_t>(out);
}

heap::heap(live_heap &&h)
{
    assert(h.is_contiguous());
    log_debug("freezing heap with ID %d, %d item pointers, %d bytes payload",
              h.get_cnt(), h.pointers.size(), h.min_length);
    /* The length of addressed items is measured from the item to the
     * address of the next item, or the end of the heap. We may receive
     * packets (and hence pointers) out-of-order, so we have to sort.
     * The mask preserves both the address and the immediate-mode flag,
     * so that all addressed items sort together.
     *
     * The sort needs to be stable, because there can be zero-length fields.
     * TODO: these can still break if they cross packet boundaries.
     */
    item_pointer_t sort_mask =
        immediate_mask | ((item_pointer_t(1) << h.heap_address_bits) - 1);
    auto compare = [sort_mask](item_pointer_t a, item_pointer_t b) {
        return (a & sort_mask) < (b & sort_mask);
    };
    std::stable_sort(h.pointers.begin(), h.pointers.end(), compare);

    pointer_decoder decoder(h.heap_address_bits);
    // Determine how much memory is needed to store immediates
    std::size_t n_immediates = 0;
    for (std::size_t i = 0; i < h.pointers.size(); i++)
    {
        item_pointer_t pointer = h.pointers[i];
        if (decoder.is_immediate(pointer))
            n_immediates++;
    }
    // Allocate memory
    const std::size_t immediate_size = h.heap_address_bits / 8;
    const std::size_t id_size = sizeof(item_pointer_t) - immediate_size;
    if (n_immediates > 0)
        immediate_payload.reset(new uint8_t[immediate_size * n_immediates]);
    uint8_t *next_immediate = immediate_payload.get();
    items.reserve(h.pointers.size());

    for (std::size_t i = 0; i < h.pointers.size(); i++)
    {
        item new_item;
        item_pointer_t pointer = h.pointers[i];
        new_item.id = decoder.get_id(pointer);
        if (new_item.id == 0)
            continue; // just padding
        new_item.is_immediate = decoder.is_immediate(pointer);
        if (new_item.is_immediate)
        {
            new_item.ptr = next_immediate;
            new_item.length = immediate_size;
            new_item.immediate_value = decoder.get_immediate(pointer);
            item_pointer_t pointer_be = htobe<item_pointer_t>(pointer);
            std::memcpy(
                next_immediate,
                reinterpret_cast<const std::uint8_t *>(&pointer_be) + id_size,
                immediate_size);
            log_debug("Found new immediate item ID %d, value %d",
                      new_item.id, new_item.immediate_value);
            next_immediate += immediate_size;
        }
        else
        {
            s_item_pointer_t start = decoder.get_address(pointer);
            s_item_pointer_t end;
            if (i + 1 < h.pointers.size()
                && !decoder.is_immediate(h.pointers[i + 1]))
                end = decoder.get_address(h.pointers[i + 1]);
            else
                end = h.min_length;
            assert(start <= h.min_length);
            if (start == end)
            {
                log_debug("skipping empty item %d", new_item.id);
                continue;
            }
            new_item.ptr = h.payload.get() + start;
            new_item.length = end - start;
            log_debug("found new addressed item ID %d, offset %d, length %d",
                      new_item.id, start, end - start);
        }
        items.push_back(new_item);
    }
    cnt = h.cnt;
    flavour_ = flavour(maximum_version, 8 * sizeof(item_pointer_t),
                       h.heap_address_bits, h.bug_compat);
    payload = std::move(h.payload);
    // Reset h so that it still satisfies its invariants
    h = live_heap(0, h.bug_compat, h.allocator);
}

descriptor heap::to_descriptor() const
{
    const std::size_t immediate_size = flavour_.get_heap_address_bits() / 8;
    const std::size_t id_size = sizeof(item_pointer_t) - immediate_size;
    descriptor out;
    for (const item &item : items)
    {
        switch (item.id)
        {
            case DESCRIPTOR_ID_ID:
                if (item.is_immediate)
                    out.id = load_bytes_be(item.ptr, item.length);
                else
                    log_info("Ignoring descriptor ID that is not an immediate");
                break;
            case DESCRIPTOR_NAME_ID:
                out.name = std::string(reinterpret_cast<const char *>(item.ptr), item.length);
                break;
            case DESCRIPTOR_DESCRIPTION_ID:
                out.description = std::string(reinterpret_cast<const char *>(item.ptr), item.length);
                break;
            case DESCRIPTOR_FORMAT_ID:
                {
                    int field_size = (flavour_.get_bug_compat() & BUG_COMPAT_DESCRIPTOR_WIDTHS) ? 4 : 1 + id_size;
                    for (std::size_t i = 0; i + field_size <= item.length; i += field_size)
                    {
                        char type = item.ptr[i];
                        std::int64_t bits = load_bytes_be(item.ptr + i + 1, field_size - 1);
                        out.format.emplace_back(type, bits);
                    }
                    break;
                }
            case DESCRIPTOR_SHAPE_ID:
                {
                    int field_size = (flavour_.get_bug_compat() & BUG_COMPAT_DESCRIPTOR_WIDTHS) ? 8 : 1 + immediate_size;
                    for (std::size_t i = 0; i + field_size <= item.length; i += field_size)
                    {
                        int mask = (flavour_.get_bug_compat() & BUG_COMPAT_SHAPE_BIT_1) ? 2 : 1;
                        bool variable = (item.ptr[i] & mask);
                        std::int64_t size = variable ? -1 : load_bytes_be(item.ptr + i + 1, field_size - 1);
                        out.shape.push_back(size);
                    }
                    break;
                }
            case DESCRIPTOR_DTYPE_ID:
                out.numpy_header = std::string(reinterpret_cast<const char *>(item.ptr), item.length);
                break;
            default:
                log_info("Unrecognised descriptor item ID %#x", item.id);
                break;
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

class descriptor_stream : public stream_base
{
private:
    virtual void heap_ready(live_heap &&h) override;
public:
    using stream_base::stream_base;
    std::vector<descriptor> descriptors;
};

void descriptor_stream::heap_ready(live_heap &&h)
{
    if (h.is_contiguous())
    {
        heap frozen(std::move(h));
        descriptor d = frozen.to_descriptor();
        if (d.id != 0) // check that we got an ID field
            descriptors.push_back(std::move(d));
        else
            log_info("incomplete descriptor (no ID)");
    }
}

} // anonymous namespace

std::vector<descriptor> heap::get_descriptors() const
{
    descriptor_stream s(flavour_.get_bug_compat(), 1);
    for (const item &item : items)
    {
        if (item.id == DESCRIPTOR_ID)
        {
            mem_to_stream(s, item.ptr, item.length);
            s.flush();
        }
    }
    s.stop_received();
    return s.descriptors;
}

bool heap::is_start_of_stream() const
{
    for (const item &item : items)
        if (item.id == STREAM_CTRL_ID)
        {
            item_pointer_t value = load_bytes_be(item.ptr, item.length);
            if (value == CTRL_STREAM_START)
                return true;
        }
    return false;
}

} // namespace recv
} // namespace spead2
