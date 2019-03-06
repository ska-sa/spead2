/* Copyright 2015, 2019 SKA South Africa
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
    assert(0 <= len && len <= int(sizeof(item_pointer_t)));
    item_pointer_t out = 0;
    std::memcpy(reinterpret_cast<char *>(&out) + sizeof(item_pointer_t) - len, ptr, len);
    return betoh<item_pointer_t>(out);
}

void heap_base::transfer_immediates(heap_base &&other) noexcept
{
    if (!immediate_payload)
    {
        std::memcpy(immediate_payload_inline, other.immediate_payload_inline,
                    sizeof(immediate_payload_inline));
        for (item &it : items)
            if (it.is_immediate)
                it.ptr = immediate_payload_inline + (it.ptr - other.immediate_payload_inline);
    }
}

heap_base::heap_base(heap_base &&other) noexcept
    : cnt(std::move(other.cnt)),
    flavour_(std::move(other.flavour_)),
    items(std::move(other.items)),
    immediate_payload(std::move(other.immediate_payload)),
    payload(std::move(other.payload))
{
    transfer_immediates(std::move(other));
}

heap_base &heap_base::operator=(heap_base &&other) noexcept
{
    cnt = std::move(other.cnt);
    flavour_ = std::move(other.flavour_);
    items = std::move(other.items);
    immediate_payload = std::move(other.immediate_payload);
    payload = std::move(other.payload);
    transfer_immediates(std::move(other));
    return *this;
}

void heap_base::load(live_heap &&h, bool keep_addressed, bool keep_payload)
{
    assert(h.is_contiguous() || !keep_addressed);
    item_pointer_t *first = h.pointers_begin();
    item_pointer_t *last = h.pointers_end();
    log_debug("freezing heap with ID %d, %d item pointers, %d bytes payload",
              h.get_cnt(), last - first, h.min_length);
    /* The length of addressed items is measured from the item to the
     * address of the next item, or the end of the heap. We may receive
     * packets (and hence pointers) out-of-order, so we have to sort.
     * The mask preserves both the address and the immediate-mode flag,
     * so that all addressed items sort together.
     *
     * The sort needs to be stable, because there can be zero-length fields.
     * TODO: these can still break if they cross packet boundaries.
     */
    const pointer_decoder &decoder = h.decoder;
    item_pointer_t sort_mask =
        immediate_mask | ((item_pointer_t(1) << decoder.address_bits()) - 1);
    auto compare = [sort_mask](item_pointer_t a, item_pointer_t b) {
        return (a & sort_mask) < (b & sort_mask);
    };
    std::stable_sort(first, last, compare);

    /* Determine how much memory is needed to store immediates
     * (conservative - also counts null items).
     */
    std::size_t n_immediates = 0;
    for (auto ptr = first; ptr != last; ++ptr)
    {
        if (decoder.is_immediate(*ptr))
            n_immediates++;
    }
    // Allocate memory if necessary
    const std::size_t immediate_size = decoder.address_bits() / 8;
    const std::size_t id_size = sizeof(item_pointer_t) - immediate_size;
    uint8_t *next_immediate;
    if (immediate_size * n_immediates > sizeof(immediate_payload_inline))
    {
        next_immediate = new uint8_t[immediate_size * n_immediates];
        immediate_payload.reset(next_immediate);
    }
    else
        next_immediate = immediate_payload_inline;
    items.reserve(keep_addressed ? last - first : n_immediates);

    for (auto ptr = first; ptr != last; ++ptr)
    {
        item new_item;
        item_pointer_t pointer = *ptr;
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
            if (!keep_addressed)
                continue;
            s_item_pointer_t start = decoder.get_address(pointer);
            s_item_pointer_t end;
            if (ptr + 1 < last && !decoder.is_immediate(ptr[1]))
                end = decoder.get_address(ptr[1]);
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
                       decoder.address_bits(), h.bug_compat);
    if (keep_payload)
        payload = std::move(h.payload);
}

bool heap_base::is_ctrl_item(ctrl_mode value) const
{
    for (const item &item : items)
        if (item.id == STREAM_CTRL_ID)
        {
            item_pointer_t item_value = load_bytes_be(item.ptr, item.length);
            if (item_value == value)
                return true;
        }
    return false;
}

bool heap_base::is_start_of_stream() const
{
    return is_ctrl_item(CTRL_STREAM_START);
}

bool heap_base::is_end_of_stream() const
{
    return is_ctrl_item(CTRL_STREAM_STOP);
}


heap::heap(live_heap &&h)
{
    assert(h.is_contiguous());
    load(std::move(h), true, true);
    // Reset h so that it still satisfies its invariants
    h.reset();
}

descriptor heap::to_descriptor() const
{
    const std::size_t immediate_size = get_flavour().get_heap_address_bits() / 8;
    const std::size_t id_size = sizeof(item_pointer_t) - immediate_size;
    bug_compat_mask bug_compat = get_flavour().get_bug_compat();
    descriptor out;
    for (const item &item : get_items())
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
                    int field_size = (bug_compat & BUG_COMPAT_DESCRIPTOR_WIDTHS) ? 4 : 1 + id_size;
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
                    int field_size = (bug_compat & BUG_COMPAT_DESCRIPTOR_WIDTHS) ? 8 : 1 + immediate_size;
                    for (std::size_t i = 0; i + field_size <= item.length; i += field_size)
                    {
                        int mask = (bug_compat & BUG_COMPAT_SHAPE_BIT_1) ? 2 : 1;
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
    descriptor_stream s(get_flavour().get_bug_compat(), 1);
    for (const item &item : get_items())
    {
        if (item.id == DESCRIPTOR_ID)
        {
            mem_to_stream(s, item.ptr, item.length);
            s.flush();
        }
    }
    s.stop();
    return s.descriptors;
}


incomplete_heap::incomplete_heap(live_heap &&h, bool keep_payload, bool keep_payload_ranges)
    : heap_length(h.heap_length), received_length(h.received_length)
{
    load(std::move(h), false, keep_payload);
    if (keep_payload_ranges)
        payload_ranges = std::move(h.payload_ranges);
    // Reset h so that it still satisfies its invariants
    h.reset();
}

std::vector<std::pair<s_item_pointer_t, s_item_pointer_t>>
incomplete_heap::get_payload_ranges() const
{
    return std::vector<std::pair<s_item_pointer_t, s_item_pointer_t>>(
        payload_ranges.begin(), payload_ranges.end());
}

} // namespace recv
} // namespace spead2
