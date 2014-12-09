#include <iostream>
#include <utility>
#include <boost/asio.hpp>
#include <chrono>
#include "udp_in.h"
#include "receiver.h"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

static time_point start = std::chrono::high_resolution_clock::now();

static void heap_callback(spead::in::heap &&heap)
{
    std::cout << "Received heap with CNT " << heap.cnt() << "; complete: " << heap.is_complete() << '\n';
    if (heap.is_contiguous())
    {
        spead::in::frozen_heap fheap(std::move(heap));
        const auto &items = fheap.get_items();
        std::cout << items.size() << " item(s)\n";
        for (const auto &item : items)
        {
            std::cout << "    ID: 0x" << std::hex << item.id << std::dec << ' ';
            if (item.is_immediate)
                std::cout << item.value.immediate;
            else
                std::cout << "[" << item.value.address.length << " bytes]";
            std::cout << '\n';
        }
        std::vector<spead::descriptor> descriptors = fheap.get_descriptors();
        for (const auto &descriptor : descriptors)
        {
            std::cout
                << "    0x" << std::hex << descriptor.id << std::dec << ":\n"
                << "        NAME:  " << descriptor.name << "\n"
                << "        DESC:  " << descriptor.description << "\n";
            if (descriptor.dtype.empty())
            {
                std::cout << "        TYPE:  ";
                for (const auto &field : descriptor.format)
                    std::cout << field.first << field.second << ",";
                std::cout << "\n";
                std::cout << "        SHAPE: ";
                for (const auto &field : descriptor.shape)
                    if (field.first)
                        std::cout << "?,";
                    else
                        std::cout << field.second << ",";
                std::cout << "\n";
            }
            else
                std::cout << "        DTYPE: " << descriptor.dtype << "\n";
        }
        time_point now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start;
        std::cout << elapsed.count() << "\n";
        std::cout << std::flush;
    }
}

int main()
{
    spead::in::receiver receiver;
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::loopback(), 8888);
    std::unique_ptr<spead::in::udp_stream> stream(
        new spead::in::udp_stream(
            receiver.get_io_service(),
            endpoint, spead::in::udp_stream::default_max_size, 8192 * 1024));
    stream->set_callback(heap_callback);
    receiver.add_stream(std::move(stream));
    receiver();
    return 0;
}
