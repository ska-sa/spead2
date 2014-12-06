#ifndef SPEAD_DEFINES_H
#define SPEAD_DEFINES_H

#include <cstdint>

namespace spead
{

static constexpr std::uint64_t immediate_mask = (std::uint64_t(1) << 63);
static constexpr std::uint16_t magic_version = 0x5304;  // 0x53 is the magic, 4 is the version

enum item_id : unsigned int
{
    HEAP_CNT_ID = 1,
    HEAP_LENGTH_ID = 2,
    PAYLOAD_OFFSET_ID = 3,
    PAYLOAD_LENGTH_ID = 4,
    DESCRIPTOR_ID = 5,
    STREAM_CTRL_ID = 6
};

} // namespace spead

#endif // SPEAD_DEFINES_H
