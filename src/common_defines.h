#ifndef SPEAD_COMMON_DEFINES_H
#define SPEAD_COMMON_DEFINES_H

#include <cstdint>
#include <vector>
#include <utility>
#include <string>
#include <unordered_map>

// Descriptors encoded as 64-40 in all flavours
#define WORKAROUND_SR_96 1

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

struct descriptor
{
    std::int64_t id = 0;
    std::string name;
    std::string description;
    std::vector<std::pair<char, std::int64_t> > format;
    std::vector<std::pair<bool, std::int64_t> > shape;
    std::string dtype;
};

// Maps IDs to descriptor information
typedef std::unordered_map<std::int64_t, descriptor> descriptor_map;

} // namespace spead

#endif // SPEAD_COMMON_DEFINES_H
