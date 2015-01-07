#include <iostream>
#include <utility>
#include <chrono>
#include <boost/asio.hpp>
#include "recv_receiver.h"
#include "recv_udp.h"
#include "recv_frozen_heap.h"
#include "recv_heap.h"
#include "recv_ring_stream.h"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

static time_point start = std::chrono::high_resolution_clock::now();

class trivial_stream : public spead::recv::stream
{
private:
    virtual void heap_ready(spead::recv::heap &&heap) override
    {
        std::cout << "Got heap " << heap.cnt();
        if (heap.is_complete())
            std::cout << " [complete]\n";
        else if (heap.is_contiguous())
            std::cout << " [contiguous]\n";
        else
            std::cout << " [incomplete]\n";
    }
public:
    using spead::recv::stream::stream;
};

void show_heap(const spead::recv::frozen_heap &fheap)
{
    std::cout << "Received heap with CNT " << fheap.cnt() << '\n';
    const auto &items = fheap.get_items();
    std::cout << items.size() << " item(s)\n";
    for (const auto &item : items)
    {
        std::cout << "    ID: 0x" << std::hex << item.id << std::dec << ' ';
        std::cout << "[" << item.length << " bytes]";
        std::cout << '\n';
    }
    std::vector<spead::descriptor> descriptors = fheap.get_descriptors();
    for (const auto &descriptor : descriptors)
    {
        std::cout
            << "    0x" << std::hex << descriptor.id << std::dec << ":\n"
            << "        NAME:  " << descriptor.name << "\n"
            << "        DESC:  " << descriptor.description << "\n";
        if (descriptor.numpy_header.empty())
        {
            std::cout << "        TYPE:  ";
            for (const auto &field : descriptor.format)
                std::cout << field.first << field.second << ",";
            std::cout << "\n";
            std::cout << "        SHAPE: ";
            for (const auto &size : descriptor.shape)
                if (size == -1)
                    std::cout << "?,";
                else
                    std::cout << size << ",";
            std::cout << "\n";
        }
        else
            std::cout << "        DTYPE: " << descriptor.numpy_header << "\n";
    }
    time_point now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start;
    std::cout << elapsed.count() << "\n";
    std::cout << std::flush;
}

static void run_trivial()
{
    trivial_stream stream;

    spead::recv::receiver receiver;
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::any(), 8888);
    receiver.emplace_reader<spead::recv::udp_reader>(
        stream,
        endpoint, spead::recv::udp_reader::default_max_size, 1024 * 1024);
    receiver();
    receiver.stop();
}

static void run_ringbuffered()
{
    spead::mempool pool(16384, 26214400, 8, 4);
    spead::recv::ring_stream<spead::ringbuffer_fd<spead::recv::heap> > stream;
    stream.set_mempool(&pool);
    spead::recv::receiver receiver;
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::any(), 8888);
    receiver.emplace_reader<spead::recv::udp_reader>(
        stream,
        endpoint, spead::recv::udp_reader::default_max_size, 8 * 1024 * 1024);
    receiver.start(1);
    while (true)
    {
        try
        {
            spead::recv::frozen_heap fh = stream.pop();
            show_heap(fh);
        }
        catch (spead::ringbuffer_stopped &e)
        {
            break;
        }
    }
}

int main()
{
    // run_trivial();
    run_ringbuffered();
    return 0;
}
