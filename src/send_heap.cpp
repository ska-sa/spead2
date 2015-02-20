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

// The descriptor is appended to @a out
static void encode_descriptor(
    const descriptor &d, int heap_address_bits, bug_compat_mask bug_compat,
    std::vector<std::uint8_t> &out)
{
    std::vector<std::uint8_t> format_store;
    std::vector<std::uint8_t> shape_store;
    std::vector<item> items;
    items.resize(5);

    if (d.id <= 0 || d.id >= (std::int64_t(1) << (63 - heap_address_bits)))
        throw std::invalid_argument("Item ID out of range");
    items.emplace_back(DESCRIPTOR_ID_ID, d.id);
    items.emplace_back(DESCRIPTOR_NAME_ID, d.name);
    items.emplace_back(DESCRIPTOR_DESCRIPTION_ID, d.description);

    const int field_size = (bug_compat & BUG_COMPAT_DESCRIPTOR_WIDTHS) ? 4 : 9 - heap_address_bits / 8;
    format_store.resize(field_size * d.format.size());
    std::size_t pos = 0;
    for (const auto &field : d.format)
    {
        format_store[pos] = field.first;
        // TODO: validate that it fits
        store_bytes_be(&format_store[pos + 1], field_size - 1, field.second);
        pos += field_size;
    }

    const int shape_size = (bug_compat & BUG_COMPAT_DESCRIPTOR_WIDTHS) ? 8 : 1 + heap_address_bits / 8;
    const std::uint8_t variable_tag = (bug_compat & BUG_COMPAT_SHAPE_BIT_1) ? 2 : 1;
    shape_store.resize(shape_size * d.shape.size());
    pos = 0;
    for (const std::int64_t dim : d.shape)
    {
        if (dim < 0)
            shape_store[pos] = variable_tag;
        else
        {
            // TODO: validate that it fits
            store_bytes_be(&shape_store[pos + 1], shape_size - 1, dim);
        }
        pos += shape_size;
    }
    items.emplace_back(DESCRIPTOR_FORMAT_ID, format_store);
    items.emplace_back(DESCRIPTOR_SHAPE_ID, shape_store);
    if (!d.numpy_header.empty())
    {
        items.emplace_back(DESCRIPTOR_DTYPE_ID, d.numpy_header);
    }

    basic_heap h(1, std::move(items), nullptr);
    packet_generator gen(h, heap_address_bits, SIZE_MAX);
    const packet &pkt = gen.next_packet();
    for (const auto &buffer : pkt.buffers)
    {
        const std::uint8_t *data = boost::asio::buffer_cast<const std::uint8_t *>(buffer);
        out.insert(out.end(), data, data + buffer_size(buffer));
    }
}

basic_heap heap::encode(int heap_address_bits, bug_compat_mask bug_compat) const
{
    assert(heap_address_bits > 0 && heap_address_bits < 64 && heap_address_bits % 8 == 0);
    std::vector<item> items;
    std::unique_ptr<std::uint8_t[]> descriptor_payloads;

    if (!descriptors.empty())
    {
        std::vector<std::uint8_t> descriptor_data;
        std::vector<std::pair<std::size_t, std::size_t> > descriptor_pos; // offset, length
        descriptor_pos.reserve(descriptors.size());
        for (const descriptor &d : descriptors)
        {
            std::size_t offset = descriptor_data.size();
            encode_descriptor(d, heap_address_bits, bug_compat, descriptor_data);
            descriptor_pos.emplace_back(offset, descriptor_data.size() - offset);
        }

        // Add the items for the descriptors
        descriptor_payloads.reset(new std::uint8_t[descriptor_data.size()]);
        std::memcpy(descriptor_payloads.get(), descriptor_data.data(), descriptor_data.size());
        for (const auto &pos : descriptor_pos)
            items.emplace_back(
                DESCRIPTOR_ID,
                descriptor_payloads.get() + pos.first,
                pos.second);
    }
    items.insert(items.end(), this->items.begin(), this->items.end());

    return basic_heap(heap_cnt, std::move(items), std::move(descriptor_payloads));
}

} // namespace send
} // namespace spead
