#include <iostream>
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
    boost::asio::ip::udp::socket socket(tp.get_io_service());
    socket.connect(*it);
    spead::send::udp_stream stream(std::move(socket), 48, 7, 9000, 1e9);

    spead::send::heap h(0x2);
    std::int32_t value1 = 5;
    std::int32_t value2[64] = {};
    spead::descriptor desc1;
    desc1.id = 0x1000;
    desc1.name = "value1";
    desc1.description = "a scalar int";
    desc1.format.emplace_back('i', 32);
    spead::descriptor desc2;
    desc2.id = 0x1001;
    desc2.name = "value2";
    desc2.description = "a 2D array";
    desc2.numpy_header = "{'shape': (8, 8), 'fortran_order': False, 'descr': 'i4'}";

    h.add_item(0x1000, &value1, sizeof(value1));
    h.add_item(0x1001, &value2, sizeof(value2));
    h.add_descriptor(desc1);
    h.add_descriptor(desc2);
    stream.async_send_heap(std::move(h), [] { std::cout << "Callback fired\n"; });
    stream.flush();

    return 0;
}
