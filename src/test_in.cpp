#include <iostream>
#include <utility>
#include <boost/asio.hpp>
#include "udp_in.h"
#include "receiver.h"

static void heap_callback(spead::in::heap &&heap)
{
    std::cout << "Received heap with CNT " << heap.cnt() << "; complete: " << heap.is_complete() << '\n';
}

int main()
{
    spead::in::receiver<spead::in::udp_stream> receiver;
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::loopback(), 8888);
    spead::in::udp_stream stream(receiver, endpoint);
    stream.set_callback(heap_callback);
    receiver.add_stream(std::move(stream));
    receiver();
    return 0;
}
