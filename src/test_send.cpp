#include <iostream>
#include <utility>
#include <endian.h>
#include <boost/asio.hpp>
#include "common_thread_pool.h"
#include "common_defines.h"
#include "send_heap.h"
#include "send_udp.h"
#include "send_stream.h"

using boost::asio::ip::udp;

int main()
{
    spead::thread_pool tp;
    udp::resolver resolver(tp.get_io_service());
    udp::resolver::query query("localhost", "8888");
    auto it = resolver.resolve(query);
    spead::send::udp_stream stream(tp.get_io_service(), *it, spead::send::stream_config(48, 9000, 0));

    spead::send::heap h(0x2, 7);
    std::int32_t value1 = htobe32(0xEADBEEF);
    std::int32_t value2[64] = {};
    for (int i = 0; i < 64; i++)
        value2[i] = i;
    spead::descriptor desc1;
    desc1.id = 0x1000;
    desc1.name = "value1";
    desc1.description = "a scalar int";
    desc1.format.emplace_back('i', 32);
    spead::descriptor desc2;
    desc2.id = 0x1001;
    desc2.name = "value2";
    desc2.description = "a 2D array";
    desc2.numpy_header = "{'shape': (8, 8), 'fortran_order': False, 'descr': '>i4'}";

    h.add_item(0x1000, &value1, sizeof(value1), true);
    h.add_item(0x1001, &value2, sizeof(value2), true);
    h.add_descriptor(desc1);
    h.add_descriptor(desc2);
    stream.async_send_heap(std::move(h), [] { std::cout << "Callback fired\n"; });
    stream.flush();

    return 0;
}
