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

#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <boost/asio/buffer.hpp>
#include <spead2/send_heap.h>
#include <spead2/send_packet.h>
#include <spead2/send_utils.h>
#include <spead2/common_defines.h>
#include <spead2/common_logging.h>
#include <spead2/common_endian.h>

namespace spead2
{
namespace send
{

/**
 * Encode @a value as an unsigned big-endian number in @a len bytes.
 *
 * @pre
 * - @a 0 &lt;= len &lt;= sizeof(item_pointer_t)
 * - @a value &lt; 2<sup>len * 8</sup>
 */
static inline void store_bytes_be(std::uint8_t *ptr, int len, item_pointer_t value)
{
    assert(0 <= len && len <= sizeof(item_pointer_t));
    assert(len == sizeof(item_pointer_t) || value < (std::uint64_t(1) << (8 * len)));
    value = htobe<item_pointer_t>(value);
    std::memcpy(ptr, reinterpret_cast<const char *>(&value) + sizeof(item_pointer_t) - len, len);
}

/* Copies, then increments dest
 */
static inline void memcpy_adjust(std::uint8_t *&dest, const void *src, std::size_t length)
{
    std::memcpy(dest, src, length);
    dest += length;
}

static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t>
encode_descriptor(const descriptor &d, const flavour &flavour_)
{
    const int field_size = (flavour_.get_bug_compat() & BUG_COMPAT_DESCRIPTOR_WIDTHS) ? 4 : sizeof(item_pointer_t) + 1 - flavour_.get_heap_address_bits() / 8;
    const int shape_size = (flavour_.get_bug_compat() & BUG_COMPAT_DESCRIPTOR_WIDTHS) ? 8 : 1 + flavour_.get_heap_address_bits() / 8;

    if (d.id <= 0 || d.id >= (s_item_pointer_t(1) << (8 * sizeof(item_pointer_t) - 1 - flavour_.get_heap_address_bits())))
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
    std::size_t total_size = payload_size + n_items * sizeof(item_pointer_t) + 8;
    std::unique_ptr<std::uint8_t[]> out(new std::uint8_t[total_size]);
    std::uint64_t *header = reinterpret_cast<std::uint64_t *>(out.get());
    item_pointer_t *pointer = reinterpret_cast<item_pointer_t *>(out.get() + 8);
    std::size_t offset = 0;
    // TODO: for >64-bit item pointers, this will have alignment issues

    pointer_encoder encoder(flavour_.get_heap_address_bits());
    *header = htobe<std::uint64_t>(
            (std::uint64_t(0x5304) << 48)
            | (std::uint64_t(8 - flavour_.get_heap_address_bits() / 8) << 40)
            | (std::uint64_t(flavour_.get_heap_address_bits() / 8) << 32)
            | n_items);
    *pointer++ = htobe<item_pointer_t>(encoder.encode_immediate(HEAP_CNT_ID, 1));
    *pointer++ = htobe<item_pointer_t>(encoder.encode_immediate(HEAP_LENGTH_ID, payload_size));
    *pointer++ = htobe<item_pointer_t>(encoder.encode_immediate(PAYLOAD_OFFSET_ID, 0));
    *pointer++ = htobe<item_pointer_t>(encoder.encode_immediate(PAYLOAD_LENGTH_ID, payload_size));
    *pointer++ = htobe<item_pointer_t>(encoder.encode_immediate(DESCRIPTOR_ID_ID, d.id));
    *pointer++ = htobe<item_pointer_t>(encoder.encode_address(DESCRIPTOR_NAME_ID, offset));
    offset += d.name.size();
    *pointer++ = htobe<item_pointer_t>(encoder.encode_address(DESCRIPTOR_DESCRIPTION_ID, offset));
    offset += d.description.size();
    *pointer++ = htobe<item_pointer_t>(encoder.encode_address(DESCRIPTOR_FORMAT_ID, offset));
    offset += d.format.size() * field_size;
    *pointer++ = htobe<item_pointer_t>(encoder.encode_address(DESCRIPTOR_SHAPE_ID, offset));
    offset += d.shape.size() * shape_size;
    if (have_numpy)
    {
        *pointer++ = htobe<item_pointer_t>(encoder.encode_address(DESCRIPTOR_DTYPE_ID, offset));
        offset += d.numpy_header.size();
    }
    assert(offset == payload_size);

    std::uint8_t *data = reinterpret_cast<std::uint8_t *>(pointer);
    memcpy_adjust(data, d.name.data(), d.name.size());
    memcpy_adjust(data, d.description.data(), d.description.size());

    for (const auto &field : d.format)
    {
        *data = field.first;
        // TODO: validate that it fits
        store_bytes_be(data + 1, field_size - 1, field.second);
        data += field_size;
    }

    const std::uint8_t variable_tag = (flavour_.get_bug_compat() & BUG_COMPAT_SHAPE_BIT_1) ? 2 : 1;
    for (const s_item_pointer_t dim : d.shape)
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


heap::heap(const flavour &flavour_)
    : flavour_(flavour_)
{
}

void heap::add_descriptor(const descriptor &descriptor)
{
    auto blob = encode_descriptor(descriptor, flavour_);
    items.emplace_back(DESCRIPTOR_ID, blob.first.get(), blob.second, false);
    storage.emplace_back(std::move(blob.first));
}

} // namespace send
} // namespace spead2
