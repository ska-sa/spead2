/**
 * @file
 */

#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <boost/asio/buffer.hpp>
#include "send_heap.h"
#include "send_packet.h"
#include "send_utils.h"
#include "common_defines.h"

namespace spead
{
namespace send
{

/**
 * Encode @a value as an unsigned big-endian number in @a len bytes.
 *
 * @pre
 * - @a 0 &lt;= len &lt;= 8
 * - @a value &lt; 2<sup>len * 8</sup>
 */
static inline void store_bytes_be(std::uint8_t *ptr, int len, std::uint64_t value)
{
    assert(0 <= len && len <= 8);
    assert(len == 8 || value < (std::uint64_t(1) << (8 * len)));
    value = htobe64(value);
    std::memcpy(ptr, reinterpret_cast<const char *>(&value) + 8 - len, len);
}

/* Copies, then increments dest
 */
static inline void memcpy_adjust(std::uint8_t *&dest, const void *src, std::size_t length)
{
    std::memcpy(dest, src, length);
    dest += length;
}

static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t>
encode_descriptor(const descriptor &d, int heap_address_bits, bug_compat_mask bug_compat)
{
    const int field_size = (bug_compat & BUG_COMPAT_DESCRIPTOR_WIDTHS) ? 4 : 9 - heap_address_bits / 8;
    const int shape_size = (bug_compat & BUG_COMPAT_DESCRIPTOR_WIDTHS) ? 8 : 1 + heap_address_bits / 8;

    if (d.id <= 0 || d.id >= (std::int64_t(1) << (63 - heap_address_bits)))
        throw std::invalid_argument("Item ID out of range");

    /* The descriptor is a complete SPEAD packet, containing:
     * - header
     * - heap cnt, payload offset, payload size, heap size
     * - ID, name, description, format, shape
     * - optionally, numpy_header
     */
    bool have_numpy = !d.numpy_header.empty();
    int n_items = 9 + have_numpy;
    std::size_t payload_size =
        d.name.size()
        + d.description.size()
        + d.format.size() * field_size
        + d.shape.size() * shape_size
        + d.numpy_header.size();
    std::size_t total_size = payload_size + 8 * (n_items + 1);
    std::unique_ptr<std::uint8_t[]> out(new std::uint8_t[total_size]);
    std::uint64_t *header = reinterpret_cast<std::uint64_t *>(out.get());
    std::size_t offset = 0;

    pointer_encoder encoder(heap_address_bits);
    *header++ = htobe64(
            (std::uint64_t(0x5304) << 48)
            | (std::uint64_t(8 - heap_address_bits / 8) << 40)
            | (std::uint64_t(heap_address_bits / 8) << 32)
            | n_items);
    *header++ = htobe64(encoder.encode_immediate(HEAP_CNT_ID, 1));
    *header++ = htobe64(encoder.encode_immediate(HEAP_LENGTH_ID, payload_size));
    *header++ = htobe64(encoder.encode_immediate(PAYLOAD_OFFSET_ID, 0));
    *header++ = htobe64(encoder.encode_immediate(PAYLOAD_LENGTH_ID, payload_size));
    *header++ = htobe64(encoder.encode_immediate(DESCRIPTOR_ID_ID, d.id));
    *header++ = htobe64(encoder.encode_address(DESCRIPTOR_NAME_ID, offset));
    offset += d.name.size();
    *header++ = htobe64(encoder.encode_address(DESCRIPTOR_DESCRIPTION_ID, offset));
    offset += d.description.size();
    *header++ = htobe64(encoder.encode_address(DESCRIPTOR_FORMAT_ID, offset));
    offset += d.format.size() * field_size;
    *header++ = htobe64(encoder.encode_address(DESCRIPTOR_SHAPE_ID, offset));
    offset += d.shape.size() * shape_size;
    if (have_numpy)
    {
        *header++ = htobe64(encoder.encode_address(DESCRIPTOR_DTYPE_ID, offset));
        offset += d.numpy_header.size();
    }
    assert(offset == payload_size);

    std::uint8_t *data = reinterpret_cast<std::uint8_t *>(header);
    memcpy_adjust(data, d.name.data(), d.name.size());
    memcpy_adjust(data, d.description.data(), d.description.size());

    for (const auto &field : d.format)
    {
        *data = field.first;
        // TODO: validate that it fits
        store_bytes_be(data + 1, field_size - 1, field.second);
        data += field_size;
    }

    const std::uint8_t variable_tag = (bug_compat & BUG_COMPAT_SHAPE_BIT_1) ? 2 : 1;
    for (const std::int64_t dim : d.shape)
    {
        *data = (dim < 0) ? variable_tag : 0;
        // TODO: validate that it fits
        store_bytes_be(data + 1, shape_size - 1, dim < 0 ? 0 : dim);
        data += shape_size;
    }
    if (have_numpy)
    {
        memcpy_adjust(data, d.numpy_header.data(), d.numpy_header.size());
    }
    assert(std::size_t(data - out.get()) == total_size);
    return {std::move(out), total_size};
}

basic_heap heap::encode(int heap_address_bits, bug_compat_mask bug_compat) const
{
    assert(heap_address_bits > 0 && heap_address_bits < 64 && heap_address_bits % 8 == 0);
    std::vector<item> items;
    std::vector<std::unique_ptr<std::uint8_t[]> > descriptor_pointers;

    for (const descriptor &d : descriptors)
    {
        auto blob = encode_descriptor(d, heap_address_bits, bug_compat);
        items.emplace_back(DESCRIPTOR_ID, blob.first.get(), blob.second);
        descriptor_pointers.push_back(std::move(blob.first));
    }

    items.insert(items.end(), this->items.begin(), this->items.end());
    return basic_heap(heap_cnt, std::move(items), std::move(descriptor_pointers));
}

} // namespace send
} // namespace spead
