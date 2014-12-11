/**
 * @file
 */

#include <cassert>
#include <endian.h>
#include "recv_packet.h"
#include "recv_utils.h"
#include "common_defines.h"

namespace spead
{
namespace recv
{

/**
 * Retrieve bits [first, first+cnt) from a 64-bit field.
 *
 * @pre 0 &lt;= @a first &lt; @a first + @a cnt &lt;= 64 and @a cnt &lt; 64.
 */
static inline std::uint64_t extract_bits(std::uint64_t value, int first, int cnt)
{
    assert(0 <= first && first + cnt <= 64 && cnt > 0 && cnt < 64);
    return (value >> first) & ((std::uint64_t(1) << cnt) - 1);
}

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
    // Load for special items
    pointer_decoder decoder(heap_address_bits);
    for (int i = 1; i <= out.n_items; i++)
    {
        uint64_t pointer = be64toh(data64[i]);
        if (decoder.is_immediate(pointer))
        {
            switch (decoder.get_id(pointer))
            {
            case HEAP_CNT_ID:
                out.heap_cnt = decoder.get_immediate(pointer);
                break;
            case HEAP_LENGTH_ID:
                out.heap_length = decoder.get_immediate(pointer);
                break;
            case PAYLOAD_OFFSET_ID:
                out.payload_offset = decoder.get_immediate(pointer);
                break;
            case PAYLOAD_LENGTH_ID:
                out.payload_length = decoder.get_immediate(pointer);
                break;
            default:
                break;
            }
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

} // namespace recv
} // namespace spead

