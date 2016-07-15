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

#include <iostream>
#include <utility>
#include <boost/asio.hpp>
#include <spead2/common_endian.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_defines.h>
#include <spead2/common_flavour.h>
#include <spead2/send_heap.h>
#include <spead2/send_udp.h>
#include <spead2/send_stream.h>

using boost::asio::ip::udp;

int main()
{
    spead2::thread_pool tp;
    udp::resolver resolver(tp.get_io_service());
    udp::resolver::query query("127.0.0.1", "8888");
    auto it = resolver.resolve(query);
    spead2::send::udp_stream stream(tp.get_io_service(), *it, spead2::send::stream_config(9000, 0));
    spead2::flavour f(spead2::maximum_version, 64, 48, spead2::BUG_COMPAT_PYSPEAD_0_5_2);

    spead2::send::heap h(f);
    std::int32_t value1 = spead2::htobe(std::uint32_t(0xEADBEEF));
    std::int32_t value2[64] = {};
    for (int i = 0; i < 64; i++)
        value2[i] = i;
    spead2::descriptor desc1;
    desc1.id = 0x1000;
    desc1.name = "value1";
    desc1.description = "a scalar int";
    desc1.format.emplace_back('i', 32);
    spead2::descriptor desc2;
    desc2.id = 0x1001;
    desc2.name = "value2";
    desc2.description = "a 2D array";
    desc2.numpy_header = "{'shape': (8, 8), 'fortran_order': False, 'descr': '>i4'}";

    h.add_item(0x1000, &value1, sizeof(value1), true);
    h.add_item(0x1001, &value2, sizeof(value2), true);
    h.add_descriptor(desc1);
    h.add_descriptor(desc2);
    stream.async_send_heap(h, [] (const boost::system::error_code &ec, spead2::item_pointer_t bytes_transferred)
    {
        if (ec)
            std::cerr << ec.message() << '\n';
        else
            std::cout << "Sent " << bytes_transferred << " bytes in heap\n";
    });

    spead2::send::heap end(f);
    end.add_end();
    stream.async_send_heap(end, [] (const boost::system::error_code &ec, spead2::item_pointer_t bytes_transferred) {});
    stream.flush();

    return 0;
}
