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

#ifndef SPEAD2_COMMON_RAW_PACKET_H
#define SPEAD2_COMMON_RAW_PACKET_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <array>
#include <boost/preprocessor/cat.hpp>
#include <boost/asio/ip/address_v4.hpp>
#include <boost/asio/ip/address.hpp>
#include <boost/asio/buffer.hpp>
#include <spead2/common_endian.h>

namespace spead2
{

/// An ethernet MAC address.
typedef std::array<std::uint8_t, 6> mac_address;

/**
 * Return the MAC address corresponding to an IPv4 multicast group, as
 * defined in RFC 7042.
 */
mac_address multicast_mac(const boost::asio::ip::address_v4 &address);
mac_address multicast_mac(const boost::asio::ip::address &address);

/**
 * Determine the MAC address for an interface, given the interface's IP address.
 *
 * @throw std::invalid_argument if no interface with this IP address is found.
 */
mac_address interface_mac(const boost::asio::ip::address &address);

class packet_buffer
{
private:
    std::uint8_t *ptr;
    std::size_t length;

protected:
    template<typename T>
    T get(std::size_t offset) const
    {
        T out;
        std::memcpy(&out, ptr + offset, sizeof(out));
        return out;
    }

    template<typename T>
    void set(std::size_t offset, const T &value)
    {
        std::memcpy(ptr + offset, &value, sizeof(value));
    }

    template<typename T>
    T get_be(std::size_t offset) const
    {
        return betoh(get<T>(offset));
    }

    template<typename T>
    void set_be(std::size_t offset, T value)
    {
        set<T>(offset, htobe(value));
    }

public:
    packet_buffer();
    packet_buffer(void *ptr, std::size_t length);
    operator boost::asio::mutable_buffer() const;

    std::uint8_t *data() const;
    std::size_t size() const;
};

#define SPEAD2_DECLARE_FIELD(offset, type, name, transform) \
        type name() const { return BOOST_PP_CAT(get, transform)<type>(offset); } \
        void name(const type &value) { BOOST_PP_CAT(set, transform)<type>(offset, value); }

class udp_packet : public packet_buffer
{
public:
    static constexpr std::size_t min_size = 8;
    static constexpr std::uint8_t protocol = 0x11;

    udp_packet() = default;
    udp_packet(void *ptr, std::size_t size);

    SPEAD2_DECLARE_FIELD(0, std::uint16_t, source_port, _be)
    SPEAD2_DECLARE_FIELD(2, std::uint16_t, destination_port, _be)
    SPEAD2_DECLARE_FIELD(4, std::uint16_t, length, _be)
    SPEAD2_DECLARE_FIELD(6, std::uint16_t, checksum, _be)

    packet_buffer payload() const;
};

class ipv4_packet : public packet_buffer
{
private:
    template<typename T>
    T get_ip(std::size_t offset) const
    {
        return T(get<typename T::bytes_type>(offset));
    }

    template<typename T>
    void set_ip(std::size_t offset, const T &addr)
    {
        set(offset, addr.to_bytes());
    }

public:
    static constexpr std::size_t min_size = 20;
    static constexpr std::uint16_t ethertype = 0x0800;
    static constexpr std::uint16_t flag_do_not_fragment = 0x4000;
    static constexpr std::uint16_t flag_more_fragments = 0x2000;

    ipv4_packet() = default;
    ipv4_packet(void *ptr, std::size_t size);

    SPEAD2_DECLARE_FIELD(0,  std::uint8_t,  version_ihl,)
    SPEAD2_DECLARE_FIELD(1,  std::uint8_t,  dscp_ecn,)
    SPEAD2_DECLARE_FIELD(2,  std::uint16_t, total_length, _be)
    SPEAD2_DECLARE_FIELD(4,  std::uint16_t, identification, _be)
    SPEAD2_DECLARE_FIELD(6,  std::uint16_t, flags_frag_off, _be)
    SPEAD2_DECLARE_FIELD(8,  std::uint8_t,  ttl,)
    SPEAD2_DECLARE_FIELD(9,  std::uint8_t,  protocol,)
    SPEAD2_DECLARE_FIELD(10, std::uint16_t, checksum, _be)
    SPEAD2_DECLARE_FIELD(12, boost::asio::ip::address_v4, source_address, _ip)
    SPEAD2_DECLARE_FIELD(16, boost::asio::ip::address_v4, destination_address, _ip)

    void update_checksum();

    // Computed values, not raw fields
    bool is_fragment() const;
    std::size_t header_length() const;
    int version() const;

    udp_packet payload_udp() const;
};

/* Wraps a block of contiguous data and provides access to the ethernet
 * header fields. All fields are queried and set using native endian,
 * unless otherwise specified.
 */
class ethernet_frame : public packet_buffer
{
public:
    static constexpr std::size_t min_size = 14;

    ethernet_frame() = default;
    ethernet_frame(void *ptr, std::size_t size);

    SPEAD2_DECLARE_FIELD(0, mac_address, destination_mac,)
    SPEAD2_DECLARE_FIELD(6, mac_address, source_mac,)
    SPEAD2_DECLARE_FIELD(12, std::uint16_t, ethertype, _be)
    // TODO: Handle VLAN tags

    ipv4_packet payload_ipv4() const;
};

#undef SPEAD2_DECLARE_FIELD

class packet_type_error : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/**
 * Inspect an ethernet frame to extract the UDP4 payload, with sanity checks.
 *
 * @throws length_error if any length fields are invalid
 * @throws packet_type_error if there are other problems e.g. it is not an IPv4 packet
 */
packet_buffer udp_from_ethernet(void *ptr, size_t size);

} // namespace spead2

#endif // SPEAD2_COMMON_RAW_PACKET_H
