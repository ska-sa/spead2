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

/**
 * @file
 *
 * Microbenchmark for the ringbuffer implementation.
 */

#include <iostream>
#include <thread>
#include <future>
#include <type_traits>
#include <chrono>
#include <cstdint>
#include <memory>
#include <cerrno>
#include <sched.h>
#include <boost/program_options.hpp>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_semaphore.h>
#include <spead2/common_thread_pool.h>
#include <spead2/recv_heap.h>

namespace po = boost::program_options;

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;
typedef std::aligned_storage<sizeof(spead2::recv::heap), alignof(spead2::recv::heap)>::type item_t;

static constexpr std::size_t alignment = 64;

struct options
{
    std::string type = "light";
    std::size_t capacity = 2;
    std::int64_t items = 100000;
    int producer_cpu = -1;
    int consumer_cpu = -1;
};

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: test_ringbuffer [options]\n";
    o << desc;
}

template<typename T>
static po::typed_value<T> *make_opt(T &var)
{
    return po::value<T>(&var)->default_value(var);
}

static options parse_args(int argc, const char **argv)
{
    options opts;
    po::options_description desc;
    desc.add_options()
        ("type", make_opt(opts.type), "Semaphore type (light | fd | pipe | eventfd | posix)")
        ("capacity", make_opt(opts.capacity), "Ring buffer capacity")
        ("items", make_opt(opts.items), "Items to transmit")
        ("producer-cpu,p", make_opt(opts.producer_cpu), "CPU core to bind producer to")
        ("consumer-cpu,c", make_opt(opts.consumer_cpu), "CPU core to bind consumer to");

    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
            .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
            .options(desc)
            .run(), vm);
        po::notify(vm);
        if (vm.count("help"))
        {
            usage(std::cout, desc);
            std::exit(0);
        }
        return opts;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        usage(std::cerr, desc);
        std::exit(2);
    }
}

static void bind_cpu(int cpu)
{
    if (cpu != -1)
    {
        spead2::thread_pool::set_affinity(cpu);
    }
}

// Reimplementation of std::align, since GCC 4.8 doesn't implement it
static void *align(std::size_t alignment, std::size_t size, void *&ptr, std::size_t &space)
{
    std::uintptr_t value = reinterpret_cast<std::uintptr_t>(ptr);
    std::size_t adjust = alignment - (value & (alignment - 1));
    if (adjust == alignment)
        adjust = 0;
    if (size + adjust > space)
        return nullptr;
    ptr = static_cast<void *>(static_cast<char *>(ptr) + adjust);
    space -= adjust;
    return ptr;
}

template<typename Ringbuffer>
static void reader(Ringbuffer &ring, const options &opts)
{
    bind_cpu(opts.consumer_cpu);
    try
    {
        while (true)
        {
            item_t item = ring.pop();
            (void) item;
        }
    }
    catch (spead2::ringbuffer_stopped &)
    {
    }
}

template<typename Ringbuffer>
static void run(const options &opts)
{
    // Allocate ring buffer at aligned address
    char ring_storage[sizeof(Ringbuffer) + alignment];
    void *ring_ptr = ring_storage;
    std::size_t space = sizeof(ring_storage);
    Ringbuffer *ring = reinterpret_cast<Ringbuffer *>(
        align(alignment, sizeof(Ringbuffer), ring_ptr, space));
    new (ring) Ringbuffer(opts.capacity);

    std::thread thread(std::bind(reader<Ringbuffer>, std::ref(*ring), std::cref(opts)));
    // Give the thread time to get going
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    time_point start = std::chrono::high_resolution_clock::now();
    for (std::int64_t i = 0; i < opts.items; i++)
    {
        ring->push(item_t());
    }
    ring->stop();
    thread.join();
    time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_duration = end - start;
    double elapsed = elapsed_duration.count();
    std::cout << opts.items << " in " << elapsed << "s (" << opts.items / elapsed << "/s)\n";
    ring->~Ringbuffer();
}

int main(int argc, const char **argv)
{
    options opts = parse_args(argc, argv);
    bind_cpu(opts.producer_cpu);

    if (opts.type == "fd")
        run<spead2::ringbuffer<item_t, spead2::semaphore_fd, spead2::semaphore_fd>>(opts);
    else if (opts.type == "light")
        run<spead2::ringbuffer<item_t>>(opts);
    else if (opts.type == "spin")
        run<spead2::ringbuffer<item_t, spead2::semaphore_spin, spead2::semaphore_spin>>(opts);
    else if (opts.type == "pipe")
        run<spead2::ringbuffer<item_t, spead2::semaphore_pipe, spead2::semaphore_pipe>>(opts);
#if SPEAD2_USE_POSIX_SEMAPHORES
    else if (opts.type == "posix")
        run<spead2::ringbuffer<item_t, spead2::semaphore_posix, spead2::semaphore_posix>>(opts);
#endif
#if SPEAD2_USE_EVENTFD
    else if (opts.type == "eventfd")
        run<spead2::ringbuffer<item_t, spead2::semaphore_eventfd, spead2::semaphore_eventfd>>(opts);
#endif
    else
    {
        std::cerr << "Unknown semaphore type " << opts.type << "\n";
        return 2;
    }

    return 0;
}
