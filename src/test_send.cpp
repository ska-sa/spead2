#include <iostream>
#include <boost/asio.hpp>
#include "common_thread_pool.h"
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
    spead::send::udp_stream stream(std::move(socket), 40, 9000, 1e9);

    std::vector<spead::send::item> items;
    int value1 = 5;
    int value2[64] = {};
    items.emplace_back(0x1000, &value1, sizeof(value1));
    items.emplace_back(0x1001, &value2, sizeof(value2));
    spead::send::heap h(0x2, std::move(items));
    stream.async_send_heap(std::move(h), [] { std::cout << "Callback fired\n"; });
    stream.flush();

    return 0;
}
