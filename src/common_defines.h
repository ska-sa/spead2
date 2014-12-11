/**
 * @file
 *
 * Magic numbers and data structures for SPEAD.
 */

#ifndef SPEAD_COMMON_DEFINES_H
#define SPEAD_COMMON_DEFINES_H

#include <cstdint>
#include <vector>
#include <utility>
#include <string>

/// If true, descriptors are encoded as 64-40 regardless of actual flavour
#define BUG_COMPAT_DESCRIPTOR_WIDTHS 1
/// If true, bit 1 (value 2) of the first byte in a shape element indicates variable-size
#define BUG_COMPAT_SHAPE_BIT_1 1

/**
 * SPEAD protocol sending and receiving. All SPEAD-64-* flavours are
 * supported.
 */
namespace spead
{

static constexpr std::uint64_t immediate_mask = (std::uint64_t(1) << 63);
static constexpr std::uint16_t magic_version = 0x5304;  // 0x53 is the magic, 4 is the version

enum item_id : unsigned int
{
    HEAP_CNT_ID =          0x01,
    HEAP_LENGTH_ID =       0x02,
    PAYLOAD_OFFSET_ID =    0x03,
    PAYLOAD_LENGTH_ID =    0x04,
    DESCRIPTOR_ID =        0x05,
    STREAM_CTRL_ID =       0x06,

    DESCRIPTOR_NAME_ID =   0x10,
    DESCRIPTOR_DESCRIPTION_ID = 0x11,
    DESCRIPTOR_SHAPE_ID =  0x12,
    DESCRIPTOR_FORMAT_ID = 0x13,
    DESCRIPTOR_ID_ID =     0x14,
    DESCRIPTOR_DTYPE_ID =  0x15
};

/**
 * An unpacked descriptor.
 *
 * If @ref numpy_header is non-empty, it overrides @ref format and @ref shape.
 */
struct descriptor
{
    std::int64_t id = 0;
    std::string name;
    std::string description;
    /// Each element is a specifier character (e.g. 'u' for unsigned) and a bit width
    std::vector<std::pair<char, std::int64_t> > format;
    /// First element is true if the first is variable-length, otherwise false
    std::vector<std::pair<bool, std::int64_t> > shape;
    /// Description in the format used in .npy files.
    std::string numpy_header;
};

} // namespace spead

#endif // SPEAD_COMMON_DEFINES_H
