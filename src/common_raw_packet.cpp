/* Copyright 2016 SKA South Africa
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
 * Utilities for encoding and decoding ethernet, IP and UDP headers. These
 * are portable definitions that do not depend on OS-specific header files
 * or compiler-specific magic to specify mis-aligned storage.
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <array>
#include <stdexcept>
#include <boost/asio/ip/address_v4.hpp>
#include "common_raw_packet.h"
#include "common_endian.h"

namespace spead2
{

/////////////////////////////////////////////////////////////////////////////

packet_buffer::packet_buffer() : ptr(nullptr), length(0) {}

packet_buffer::packet_buffer(void *ptr, std::size_t size)
    : ptr(reinterpret_cast<unsigned char *>(ptr)), length(size)
{
}

unsigned char *packet_buffer::get() const
{
    return ptr;
}

std::size_t packet_buffer::size() const
{
    return length;
}

/////////////////////////////////////////////////////////////////////////////

constexpr std::uint8_t udp_packet::protocol;
constexpr std::size_t udp_packet::min_size;

udp_packet::udp_packet(void *ptr, std::size_t size)
    : packet_buffer(ptr, size)
{
    if (size < min_size)
        throw std::length_error("packet is too small to be a UDP packet");
}

packet_buffer udp_packet::payload() const
{
    std::size_t len = length();
    if (len > size() || len < min_size)
        throw std::length_error("length header is invalid");
    return packet_buffer(get() + min_size, length() - min_size);
}

/////////////////////////////////////////////////////////////////////////////

constexpr std::uint16_t ipv4_packet::ethertype;
constexpr std::size_t ipv4_packet::min_size;

ipv4_packet::ipv4_packet(void *ptr, std::size_t size)
    : packet_buffer(ptr, size)
{
    if (size < min_size)
        throw std::length_error("packet is too small to be an IPv4 packet");
}

bool ipv4_packet::is_fragment() const
{
    // If either the more fragments flag is set, or we have a non-zero offset
    return flags_frag_off() & (FLAG_MORE_FRAGMENTS | 0x1fff);
}

std::size_t ipv4_packet::header_length() const
{
    return 4 * (version_ihl() & 0xf);
}

int ipv4_packet::version() const
{
    return version_ihl() >> 4;
}

udp_packet ipv4_packet::payload_udp() const
{
    std::size_t h = header_length();
    std::size_t len = total_length();
    if (h > size() || h < min_size)
        throw std::length_error("ihl header is invalid");
    if (len > size() || len < h)
        throw std::length_error("length header is invalid");
    return udp_packet(get() + h, total_length() - h);
}

/////////////////////////////////////////////////////////////////////////////

constexpr std::size_t ethernet_frame::min_size;

ethernet_frame::ethernet_frame(void *ptr, std::size_t size)
    : packet_buffer(ptr, size)
{
    if (size < min_size)
        throw std::length_error("packet is too small to be an ethernet frame");
}

ipv4_packet ethernet_frame::payload_ipv4() const
{
    // TODO: handle VLAN tags
    return ipv4_packet(get() + min_size, size() - min_size);
}

} // namespace spead2
