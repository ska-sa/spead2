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
#include <string>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <boost/asio/ip/address_v4.hpp>
#include <boost/asio/ip/address.hpp>
#include <sys/types.h>
#include <sys/socket.h>
#include <net/ethernet.h>
#include <net/if_arp.h>
#include <ifaddrs.h>
#include <spead2/common_raw_packet.h>
#include <spead2/common_endian.h>

// TODO: use autoconf to figure out which headers to use
#ifdef __linux__
# include <netpacket/packet.h>
#else
# include <net/if_dl.h>
#endif

namespace spead2
{

mac_address multicast_mac(const boost::asio::ip::address_v4 &address)
{
    mac_address ans;
    auto bytes = address.to_bytes();
    std::memcpy(&ans[2], &bytes, 4);
    ans[0] = 0x01;
    ans[1] = 0x00;
    ans[2] = 0x5e;
    ans[3] &= 0x7f;
    return ans;
}

mac_address multicast_mac(const boost::asio::ip::address &address)
{
    return multicast_mac(address.to_v4());
}

namespace
{
struct freeifaddrs_deleter
{
    void operator()(ifaddrs *ifa) const { freeifaddrs(ifa); }
};
} // anonymous namespace

mac_address interface_mac(const boost::asio::ip::address &address)
{
    ifaddrs *ifap;
    if (getifaddrs(&ifap) < 0)
        throw std::system_error(errno, std::system_category(), "getifaddrs failed");
    std::unique_ptr<ifaddrs, freeifaddrs_deleter> ifap_owner(ifap);

    // Map address to an interface name
    char *if_name = nullptr;
    for (ifaddrs *cur = ifap; cur; cur = cur->ifa_next)
    {
        if (cur->ifa_addr && *(sa_family_t *) cur->ifa_addr == AF_INET && address.is_v4())
        {
            const sockaddr_in *cur_address = (const sockaddr_in *) cur->ifa_addr;
            const auto expected = address.to_v4().to_bytes();
            if (memcmp(&cur_address->sin_addr, &expected, sizeof(expected)) == 0)
            {
                if_name = cur->ifa_name;
                break;
            }
        }
        else if (cur->ifa_addr && *(sa_family_t *) cur->ifa_addr == AF_INET6 && address.is_v6())
        {
            const sockaddr_in6 *cur_address = (const sockaddr_in6 *) cur->ifa_addr;
            const auto expected = address.to_v6().to_bytes();
            if (memcmp(&cur_address->sin6_addr, &expected, sizeof(expected)) == 0)
            {
                if_name = cur->ifa_name;
                break;
            }
        }
    }
    if (!if_name)
    {
        throw std::runtime_error("no interface found with the address " + address.to_string());
    }

    // Now find the MAC address for this interface
    for (ifaddrs *cur = ifap; cur; cur = cur->ifa_next)
    {
#ifdef __linux__
        if (std::strcmp(cur->ifa_name, if_name) == 0
            && cur->ifa_addr && *(sa_family_t *) cur->ifa_addr == AF_PACKET)
        {
            const sockaddr_ll *ll = (sockaddr_ll *) cur->ifa_addr;
            if (ll->sll_hatype == ARPHRD_ETHER && ll->sll_halen == 6)
            {
                mac_address mac;
                std::memcpy(&mac, ll->sll_addr, 6);
                return mac;
            }
        }
#else
        if (std::strcmp(cur->ifa_name, if_name) == 0
            && cur->ifa_addr && *(sa_family_t *) cur->ifa_addr == AF_LINK)
        {
            const sockaddr_dl *dl = (sockaddr_dl *) cur->ifa_addr;
            if (dl->sdl_alen == 6)
            {
                mac_address mac;
                std::memcpy(&mac, LLADDR(dl), 6);
                return mac;
            }
        }
#endif
    }
    throw std::runtime_error(std::string("no MAC address found for interface ") + if_name);
}

/////////////////////////////////////////////////////////////////////////////

packet_buffer::packet_buffer() : ptr(nullptr), length(0) {}

packet_buffer::packet_buffer(void *ptr, std::size_t size)
    : ptr(reinterpret_cast<std::uint8_t *>(ptr)), length(size)
{
}

std::uint8_t *packet_buffer::data() const
{
    return ptr;
}

std::size_t packet_buffer::size() const
{
    return length;
}

packet_buffer::operator boost::asio::mutable_buffer() const
{
    return boost::asio::mutable_buffer(ptr, length);
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
    return packet_buffer(data() + min_size, length() - min_size);
}

/////////////////////////////////////////////////////////////////////////////

constexpr std::uint16_t ipv4_packet::ethertype;
constexpr std::size_t ipv4_packet::min_size;
constexpr std::uint16_t ipv4_packet::flag_do_not_fragment;
constexpr std::uint16_t ipv4_packet::flag_more_fragments;

ipv4_packet::ipv4_packet(void *ptr, std::size_t size)
    : packet_buffer(ptr, size)
{
    if (size < min_size)
        throw std::length_error("packet is too small to be an IPv4 packet");
}

void ipv4_packet::update_checksum()
{
    int h = header_length();
    std::uint32_t sum = 0;
    checksum(0); // avoid including the old checksum in the new
    for (int i = 0; i < h; i += 2)
        sum += get_be<std::uint16_t>(i);
    while (sum > 0xffff)
        sum = (sum & 0xffff) + (sum >> 16);
    checksum(~std::uint16_t(sum));
}

bool ipv4_packet::is_fragment() const
{
    // If either the more fragments flag is set, or we have a non-zero offset
    return flags_frag_off() & (flag_more_fragments | 0x1fff);
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
    return udp_packet(data() + h, total_length() - h);
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
    return ipv4_packet(data() + min_size, size() - min_size);
}

} // namespace spead2
