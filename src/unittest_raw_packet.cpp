/* Copyright 2019 SKA South Africa
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
 * Unit tests for common_raw_packet.
 */

#include <boost/test/unit_test.hpp>
#include <spead2/common_raw_packet.h>
#include <stdexcept>

namespace spead2
{
namespace unittest
{

// A packet captured with tcpdump, containing "Hello world\n" in the UDP payload
static const std::uint8_t sample_packet[] =
{
    0x01, 0x00, 0x5e, 0x66, 0xfe, 0x01, 0x1c, 0x1b, 0x0d, 0xe0, 0xd0, 0xfd, 0x08, 0x00, 0x45, 0x00,
    0x00, 0x28, 0xae, 0xa3, 0x40, 0x00, 0x01, 0x11, 0xd0, 0xfe, 0x0a, 0x08, 0x02, 0xb3, 0xef, 0x66,
    0xfe, 0x01, 0x87, 0x5d, 0x22, 0xb8, 0x00, 0x14, 0xe9, 0xb4, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20,
    0x77, 0x6f, 0x72, 0x6c, 0x64, 0x0a
};
// Properties of this sample packet
static const mac_address source_mac = {{0x1c, 0x1b, 0x0d, 0xe0, 0xd0, 0xfd}};
static const mac_address destination_mac = {{0x01, 0x00, 0x5e, 0x66, 0xfe, 0x01}};
static const auto source_address = boost::asio::ip::address_v4::from_string("10.8.2.179");
static const auto destination_address = boost::asio::ip::address_v4::from_string("239.102.254.1");
static const std::uint16_t source_port = 34653;
static const std::uint16_t destination_port = 8888;
static const std::uint16_t ipv4_identification = 0xaea3;
static const std::uint16_t ipv4_checksum = 0xd0fe;
static const std::uint16_t udp_checksum = 0xe9b4;
static const std::string sample_payload = "Hello world\n";

struct packet_data
{
    std::array<std::uint8_t, sizeof(sample_packet)> data;

    packet_data()
    {
        std::memcpy(data.data(), sample_packet, sizeof(sample_packet));
    }
};

BOOST_AUTO_TEST_SUITE(common)
BOOST_FIXTURE_TEST_SUITE(raw_packet, packet_data)

static std::string buffer_to_string(const boost::asio::const_buffer &buffer)
{
    return std::string(boost::asio::buffer_cast<const char *>(buffer),
                       boost::asio::buffer_cast<const char *>(buffer) + boost::asio::buffer_size(buffer));
}

static std::string buffer_to_string(const boost::asio::mutable_buffer &buffer)
{
    return buffer_to_string(boost::asio::const_buffer(buffer));
}

BOOST_AUTO_TEST_CASE(multicast_mac)
{
    auto address = boost::asio::ip::address::from_string("239.202.234.100");
    mac_address result = spead2::multicast_mac(address);
    mac_address expected = {{0x01, 0x00, 0x5e, 0x4a, 0xea, 0x64}};
    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(),
                                  expected.begin(), expected.end());
}

/* It's not really possible to unit test interface_mac because it interacts
 * closely with the OS. But we can at least check that the error paths work.
 */
BOOST_AUTO_TEST_CASE(interface_mac_lo)
{
    auto address = boost::asio::ip::address::from_string("127.0.0.1");
    BOOST_CHECK_THROW(spead2::interface_mac(address), std::runtime_error);
    address = boost::asio::ip::address::from_string("0.0.0.0");
    BOOST_CHECK_THROW(spead2::interface_mac(address), std::runtime_error);
    address = boost::asio::ip::address::from_string("::1");
    BOOST_CHECK_THROW(spead2::interface_mac(address), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(packet_buffer_construct)
{
    std::uint8_t data[2];
    packet_buffer a;
    packet_buffer b(data, 2);
    BOOST_CHECK_EQUAL(a.data(), (std::uint8_t *) nullptr);
    BOOST_CHECK_EQUAL(a.size(), 0);
    BOOST_CHECK_EQUAL(b.data(), data);
    BOOST_CHECK_EQUAL(b.size(), 2);
}

BOOST_AUTO_TEST_CASE(parse_ethernet_frame)
{
    ethernet_frame frame(data.data(), data.size());
    BOOST_CHECK(frame.source_mac() == source_mac);
    BOOST_CHECK(frame.destination_mac() == destination_mac);
    BOOST_CHECK_EQUAL(frame.ethertype(), ipv4_packet::ethertype);
}

// Just to get full test coverage
BOOST_AUTO_TEST_CASE(ethernet_frame_default)
{
    ethernet_frame();
}

BOOST_AUTO_TEST_CASE(parse_ipv4)
{
    ethernet_frame frame(data.data(), data.size());
    ipv4_packet ipv4 = frame.payload_ipv4();

    BOOST_CHECK_EQUAL(ipv4.version_ihl(), 0x45);   // version 4, header length 20
    BOOST_CHECK_EQUAL(ipv4.version(), 4);
    BOOST_CHECK_EQUAL(ipv4.dscp_ecn(), 0);
    BOOST_CHECK_EQUAL(ipv4.total_length(), 40);
    BOOST_CHECK_EQUAL(ipv4.identification(), ipv4_identification);
    BOOST_CHECK_EQUAL(ipv4.flags_frag_off(), ipv4_packet::flag_do_not_fragment);
    BOOST_CHECK_EQUAL(ipv4.ttl(), 1);
    BOOST_CHECK_EQUAL(ipv4.protocol(), udp_packet::protocol);
    BOOST_CHECK_EQUAL(ipv4.checksum(), ipv4_checksum);
    BOOST_CHECK_EQUAL(ipv4.source_address(), source_address);
    BOOST_CHECK_EQUAL(ipv4.destination_address(), destination_address);
}

BOOST_AUTO_TEST_CASE(parse_udp)
{
    ethernet_frame frame(data.data(), data.size());
    ipv4_packet ipv4 = frame.payload_ipv4();
    udp_packet udp = ipv4.payload_udp();

    BOOST_CHECK_EQUAL(udp.source_port(), source_port);
    BOOST_CHECK_EQUAL(udp.destination_port(), destination_port);
    BOOST_CHECK_EQUAL(udp.length(), 20);
    BOOST_CHECK_EQUAL(udp.checksum(), udp_checksum);
    BOOST_CHECK_EQUAL(buffer_to_string(udp.payload()), sample_payload);
}

BOOST_AUTO_TEST_CASE(ethernet_too_small)
{
    BOOST_CHECK_THROW(ethernet_frame(data.data(), 11), std::length_error);
}

BOOST_AUTO_TEST_CASE(ipv4_too_small)
{
    ethernet_frame frame(data.data(), 30);
    BOOST_CHECK_THROW(frame.payload_ipv4(), std::length_error);
}

BOOST_AUTO_TEST_CASE(udp_too_small)
{
    ethernet_frame frame(data.data(), data.size());
    ipv4_packet ipv4 = frame.payload_ipv4();
    ipv4.total_length(23);
    BOOST_CHECK_THROW(ipv4.payload_udp(), std::length_error);
}

BOOST_AUTO_TEST_CASE(udp_bad_ipv4_ihl)
{
    ethernet_frame frame(data.data(), data.size());
    ipv4_packet ipv4 = frame.payload_ipv4();
    ipv4.version_ihl(0x44);      // 16 byte header: too small
    BOOST_CHECK_THROW(ipv4.payload_udp(), std::length_error);
    ipv4.version_ihl(0x44);      // 60 byte header: bigger than total length
    BOOST_CHECK_THROW(ipv4.payload_udp(), std::length_error);
}

BOOST_AUTO_TEST_CASE(udp_bad_ipv4_total_length)
{
    ethernet_frame frame(data.data(), data.size());
    ipv4_packet ipv4 = frame.payload_ipv4();
    ipv4.total_length(1);
    BOOST_CHECK_THROW(ipv4.payload_udp(), std::length_error);
    ipv4.total_length(100);
    BOOST_CHECK_THROW(ipv4.payload_udp(), std::length_error);
}

BOOST_AUTO_TEST_CASE(udp_bad_length)
{
    ethernet_frame frame(data.data(), data.size());
    ipv4_packet ipv4 = frame.payload_ipv4();
    udp_packet udp = ipv4.payload_udp();
    udp.length(100);
    BOOST_CHECK_THROW(udp.payload(), std::length_error);
}

BOOST_AUTO_TEST_CASE(udp_from_ethernet)
{
    boost::asio::mutable_buffer payload = spead2::udp_from_ethernet(data.data(), data.size());
    BOOST_CHECK_EQUAL(buffer_to_string(payload), sample_payload);
}

BOOST_AUTO_TEST_CASE(udp_from_ethernet_not_ipv4)
{
    ethernet_frame frame(data.data(), data.size());
    ipv4_packet ipv4 = frame.payload_ipv4();
    ipv4.version_ihl(0x55);    // IPv4 ethertype, but header is not v4
    BOOST_CHECK_THROW(spead2::udp_from_ethernet(data.data(), data.size()), packet_type_error);
}

BOOST_AUTO_TEST_CASE(udp_from_ethernet_not_ipv4_ethertype)
{
    ethernet_frame frame(data.data(), data.size());
    frame.ethertype(0x86DD);   // IPv6 ethertype
    BOOST_CHECK_THROW(spead2::udp_from_ethernet(data.data(), data.size()), packet_type_error);
}

BOOST_AUTO_TEST_CASE(udp_from_ethernet_not_udp)
{
    ethernet_frame frame(data.data(), data.size());
    ipv4_packet ipv4 = frame.payload_ipv4();
    ipv4.protocol(1);          // ICMP
    BOOST_CHECK_THROW(spead2::udp_from_ethernet(data.data(), data.size()), packet_type_error);
}

BOOST_AUTO_TEST_CASE(udp_from_ethernet_fragmented)
{
    ethernet_frame frame(data.data(), data.size());
    ipv4_packet ipv4 = frame.payload_ipv4();
    ipv4.flags_frag_off(ipv4_packet::flag_more_fragments);
    BOOST_CHECK_THROW(spead2::udp_from_ethernet(data.data(), data.size()), packet_type_error);
}

// Build a packet from scratch and check that it matches the sample packet
BOOST_AUTO_TEST_CASE(build)
{
    std::array<std::uint8_t, sizeof(data)> packet = {};

    ethernet_frame frame(packet.data(), packet.size());
    frame.source_mac(source_mac);
    frame.destination_mac(destination_mac);
    frame.ethertype(ipv4_packet::ethertype);

    ipv4_packet ipv4 = frame.payload_ipv4();
    ipv4.version_ihl(0x45);        // version 4, header length 20
    ipv4.total_length(40);
    ipv4.identification(ipv4_identification);
    ipv4.flags_frag_off(ipv4_packet::flag_do_not_fragment);
    ipv4.ttl(1);
    ipv4.protocol(udp_packet::protocol);
    ipv4.source_address(source_address);
    ipv4.destination_address(destination_address);
    ipv4.update_checksum();

    udp_packet udp = ipv4.payload_udp();
    udp.source_port(source_port);
    udp.destination_port(destination_port);
    udp.length(20);
    udp.checksum(udp_checksum);

    boost::asio::mutable_buffer payload = udp.payload();
    boost::asio::buffer_copy(payload, boost::asio::buffer(sample_payload));

    BOOST_CHECK_EQUAL_COLLECTIONS(data.begin(), data.end(), packet.begin(), packet.end());
}

BOOST_AUTO_TEST_SUITE_END()  // raw_packet
BOOST_AUTO_TEST_SUITE_END()  // common

}} // namespace spead2::unittest
