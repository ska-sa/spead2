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

#ifndef SPEAD2_RECV_LIVE_HEAP_H
#define SPEAD2_RECV_LIVE_HEAP_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_set>
#include <functional>
#include "common_defines.h"
#include "common_memory_pool.h"
#include "recv_packet.h"

namespace spead2
{
namespace recv
{

class heap;

/**
 * A SPEAD heap that is in the process of being received. Once it is fully
 * received, it is converted to a @ref heap for further processing.
 *
 * Any SPEAD-64-* flavour can be used, but all packets in the heap must use
 * the same flavour. It may be possible to relax this, but it hasn't been
 * examined, and may cause issues for decoding descriptors (whose format
 * depends on the flavour).
 *
 * A heap can be:
 * - complete: a heap length item was found in a packet, and we have received
 *   all the payload corresponding to it. No more packets are expected.
 * - contiguous: the payload we have received is a contiguous range from 0 up
 *   to some amount, and cover all items described in the item pointers.
 * A complete heap is also contiguous, but not necessarily the other way
 * around. Only contiguous heaps can be frozen.
 */
class live_heap
{
private:
    friend class heap;

    /// Heap ID encoded in packets
    s_item_pointer_t cnt;
    /// Heap payload length encoded in packets (-1 for unknown)
    s_item_pointer_t heap_length = -1;
    /// Number of bytes of payload received
    s_item_pointer_t received_length = 0;
    /// True if a stream control packeting indicating end-of-heap was found
    bool end_of_stream = false;
    /**
     * Minimum possible payload size, determined from the payload range in
     * packets and item pointers, or equal to @ref heap_length if that is
     * known.
     */
    s_item_pointer_t min_length = 0;
    /// Heap address bits (from the SPEAD flavour)
    int heap_address_bits = -1;
    /// Protocol bugs to accept
    bug_compat_mask bug_compat;
    /**
     * Heap payload. When the length is unknown, this is grown by successive
     * doubling. While @c std::vector would take care of that for us, it also
     * zero-fills the memory, which would be inefficient.
     */
    memory_pool::pointer payload;
    /// Size of the memory in @ref payload
    std::size_t payload_reserved = 0;
    /**
     * Item pointers extracted from the packets, excluding those that
     * are extracted in @ref packet_header. They are in native endian.
     */
    std::vector<item_pointer_t> pointers;
    /**
     * Set of payload offsets found in packets. This is used only to
     * detect duplicate packets.
     *
     * @todo investigate more efficient structures here, e.g.
     * - a bitfield (one bit per payload byte)
     * - using a Bloom filter first (almost all queries should miss)
     * - using a linked list per offset>>13 (which is maybe equivalent to
     *   just changing the hash function)
     */
    std::unordered_set<s_item_pointer_t> packet_offsets;

    /// Backing memory pool
    std::shared_ptr<memory_pool> pool;

    /**
     * Make sure at least @a size bytes are allocated for payload. If
     * @a exact is false, then a doubling heuristic will be used.
     */
    void payload_reserve(std::size_t size, bool exact);

public:
    /**
     * Constructor.
     *
     * @param cnt          Heap ID
     * @param bug_compat   Bugs to expect in the protocol
     */
    explicit live_heap(s_item_pointer_t cnt, bug_compat_mask bug_compat);

    /**
     * Set a memory pool to use for payload data, instead of allocating with
     * @c new.
     */
    void set_memory_pool(std::shared_ptr<memory_pool> pool);

    /**
     * Attempt to add a packet to the heap. The packet must have been
     * successfully prepared by @ref decode_packet. It returns @c true if
     * the packet was added to the heap. There are a number of reasons it
     * could be rejected, even though @ref decode_packet accepted it:
     * - wrong @c cnt
     * - wrong flavour
     * - duplicate packet
     * - inconsistent heap length
     * - payload range is beyond the heap length
     */
    bool add_packet(const packet_header &packet);
    /// True if the heap is complete
    bool is_complete() const;
    /// True if the heap is contiguous
    bool is_contiguous() const;
    /// True if an end-of-stream heap control item was found
    bool is_end_of_stream() const;
    /// Retrieve the heap ID
    s_item_pointer_t get_cnt() const { return cnt; }
    /// Get protocol bug compatibility flags
    bug_compat_mask get_bug_compat() const { return bug_compat; }
};

} // namespace recv
} // namespace spead2

#endif // SPEAD2_RECV_LIVE_HEAP_H
