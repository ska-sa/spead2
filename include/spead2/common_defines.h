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
 *
 * Magic numbers and data structures for SPEAD.
 */

#ifndef SPEAD2_COMMON_DEFINES_H
#define SPEAD2_COMMON_DEFINES_H

#include <cstdint>
#include <vector>
#include <utility>
#include <string>
#include <functional>

#ifndef SPEAD2_MAX_LOG_LEVEL
#define SPEAD2_MAX_LOG_LEVEL (spead2::log_level::info)
#endif

/**
 * SPEAD protocol sending and receiving. All SPEAD-64-* flavours are
 * supported.
 */
namespace spead2
{

typedef std::uint64_t item_pointer_t;
typedef std::int64_t s_item_pointer_t;
static constexpr std::size_t item_pointer_size = sizeof(item_pointer_t);
static constexpr item_pointer_t immediate_mask = item_pointer_t(1) << (8 * item_pointer_size - 1);
static constexpr std::uint16_t magic_version = 0x5304;  // 0x53 is the magic, 4 is the version

typedef std::uint32_t bug_compat_mask;

/// Descriptors are encoded as 64-40 regardless of actual flavour
static constexpr bug_compat_mask BUG_COMPAT_DESCRIPTOR_WIDTHS = (1 << 0);
/// Bit 1 (value 2) of the first byte in a shape element indicates variable-size, instead of bit 0
static constexpr bug_compat_mask BUG_COMPAT_SHAPE_BIT_1       = (1 << 1);
/// Numpy arrays are encoded in the opposite endian to that indicated by the header
static constexpr bug_compat_mask BUG_COMPAT_SWAP_ENDIAN       = (1 << 2);
/// Bugs in PySPEAD 0.5.2
static constexpr bug_compat_mask BUG_COMPAT_PYSPEAD_0_5_2     = 0x7;

enum item_id : unsigned int
{
    NULL_ID =              0x00,
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

enum ctrl_mode : unsigned int
{
    CTRL_STREAM_START = 0,
    CTRL_DESCRIPTOR_REISSUE = 1,
    CTRL_STREAM_STOP = 2,
    CTRL_DESCRIPTOR_UPDATE = 3
};

enum memcpy_function_id : unsigned int
{
    MEMCPY_STD,
    MEMCPY_NONTEMPORAL
};

typedef std::function<void *(void * __restrict__, const void * __restrict__, std::size_t)> memcpy_function;

/**
 * An unpacked descriptor.
 *
 * If @ref numpy_header is non-empty, it overrides @ref format and @ref shape.
 */
struct descriptor
{
    /// SPEAD ID
    s_item_pointer_t id = 0;
    /// Short name
    std::string name;
    /// Long description
    std::string description;
    /**
     * Legacy format. Each element is a specifier character (e.g. 'u' for
     * unsigned) and a bit width.
     */
    std::vector<std::pair<char, s_item_pointer_t> > format;
    /**
     * Shape. Elements are either non-negative, or -1 is used to indicate a
     * variable-length size. At most one dimension may be variable-length.
     */
    std::vector<s_item_pointer_t> shape;
    /// Description in the format used in .npy files
    std::string numpy_header;
};

static constexpr int minimum_version = 4;
static constexpr int maximum_version = 4;

} // namespace spead2

#endif // SPEAD2_COMMON_DEFINES_H
