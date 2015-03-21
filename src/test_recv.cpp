#include <iostream>
#include <utility>
#include <chrono>
#include <cstdint>
#include <boost/asio.hpp>
#include "common_thread_pool.h"
#include "recv_udp.h"
#include "recv_heap.h"
#include "recv_live_heap.h"
#include "recv_ring_stream.h"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

static time_point start = std::chrono::high_resolution_clock::now();
static std::uint64_t n_complete = 0;

class trivial_stream : public spead::recv::stream
{
private:
    virtual void heap_ready(spead::recv::live_heap &&heap) override
    {
        std::cout << "Got heap " << heap.get_cnt();
        if (heap.is_complete())
        {
            std::cout << " [complete]\n";
            n_complete++;
        }
        else if (heap.is_contiguous())
            std::cout << " [contiguous]\n";
        else
            std::cout << " [incomplete]\n";
    }

    std::promise<void> stop_promise;

public:
    using spead::recv::stream::stream;

    virtual void stop_received() override
    {
        spead::recv::stream::stop_received();
        stop_promise.set_value();
    }

    void join()
    {
        std::future<void> future = stop_promise.get_future();
        future.get();
    }
};

void show_heap(const spead::recv::heap &fheap)
{
    std::cout << "Received heap with CNT " << fheap.get_cnt() << '\n';
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
    spead::thread_pool worker;
    trivial_stream stream(worker);
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::any(), 8888);
    stream.emplace_reader<spead::recv::udp_reader>(
        endpoint, spead::recv::udp_reader::default_max_size, 1024 * 1024);
    stream.join();
}

static void run_ringbuffered()
{
    spead::thread_pool worker;
    std::shared_ptr<spead::mem_pool> pool = std::make_shared<spead::mem_pool>(16384, 26214400, 12, 8);
    spead::recv::ring_stream<spead::ringbuffer_semaphore<spead::recv::live_heap> > stream(worker, 7);
    stream.set_mem_pool(pool);
    boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::any(), 8888);
    stream.emplace_reader<spead::recv::udp_reader>(
        endpoint, spead::recv::udp_reader::default_max_size, 8 * 1024 * 1024);
    while (true)
    {
        try
        {
            spead::recv::heap fh = stream.pop();
            n_complete++;
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
    std::cout << "Received " << n_complete << " complete heaps\n";
    return 0;
}
