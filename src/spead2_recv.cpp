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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/recv_udp.h>
#if SPEAD2_USE_NETMAP
# include <spead2/recv_netmap.h>
#endif
#if SPEAD2_USE_IBV
# include <spead2/recv_udp_ibv.h>
#endif
#include <spead2/recv_heap.h>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_ring_stream.h>

namespace po = boost::program_options;
namespace asio = boost::asio;

struct options
{
    bool quiet = false;
    bool descriptors = false;
    bool pyspead = false;
    bool joint = false;
    std::size_t packet = spead2::recv::udp_reader::default_max_size;
    std::string bind = "0.0.0.0";
    std::size_t buffer = spead2::recv::udp_reader::default_buffer_size;
    int threads = 1;
    std::size_t heaps = spead2::recv::stream::default_max_heaps;
    std::size_t ring_heaps = spead2::recv::ring_stream_base::default_ring_heaps;
    bool mem_pool = false;
    std::size_t mem_lower = 16384;
    std::size_t mem_upper = 32 * 1024 * 1024;
    std::size_t mem_max_free = 12;
    std::size_t mem_initial = 8;
    bool ring = false;
    bool memcpy_nt = false;
#if SPEAD2_USE_NETMAP
    std::string netmap_if;
#endif
#if SPEAD2_USE_IBV
    std::string ibv_if;
    int ibv_comp_vector = 0;
    int ibv_max_poll = spead2::recv::udp_ibv_reader::default_max_poll;
#endif
    std::vector<std::string> sources;
};

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

static time_point start = std::chrono::high_resolution_clock::now();

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: spead2_recv [options] <port>\n";
    o << desc;
}

template<typename T>
static po::typed_value<T> *make_opt(T &var)
{
    return po::value<T>(&var)->default_value(var);
}

static po::typed_value<bool> *make_opt(bool &var)
{
    return po::bool_switch(&var)->default_value(var);
}

static options parse_args(int argc, const char **argv)
{
    options opts;
    po::options_description desc, hidden, all;
    desc.add_options()
        ("quiet", make_opt(opts.quiet), "Only show total of heaps received")
        ("descriptors", make_opt(opts.descriptors), "Show descriptors")
        ("pyspead", make_opt(opts.pyspead), "Be bug-compatible with PySPEAD")
        ("joint", make_opt(opts.joint), "Treat all sources as a single stream")
        ("packet", make_opt(opts.packet), "Maximum packet size to use for UDP")
        ("bind", make_opt(opts.bind), "Bind socket to this hostname")
        ("buffer", make_opt(opts.buffer), "Socket buffer size")
        ("threads", make_opt(opts.threads), "Number of worker threads")
        ("heaps", make_opt(opts.heaps), "Maximum number of in-flight heaps")
        ("ring-heaps", make_opt(opts.ring_heaps), "Ring buffer capacity in heaps")
        ("mem-pool", make_opt(opts.mem_pool), "Use a memory pool")
        ("mem-lower", make_opt(opts.mem_lower), "Minimum allocation which will use the memory pool")
        ("mem-upper", make_opt(opts.mem_upper), "Maximum allocation which will use the memory pool")
        ("mem-max-free", make_opt(opts.mem_max_free), "Maximum free memory buffers")
        ("mem-initial", make_opt(opts.mem_initial), "Initial free memory buffers")
        ("ring", make_opt(opts.ring), "Use ringbuffer instead of callbacks")
        ("memcpy-nt", make_opt(opts.memcpy_nt), "Use non-temporal memcpy")
#if SPEAD2_USE_NETMAP
        ("netmap", make_opt(opts.netmap_if), "Netmap interface")
#endif
#if SPEAD2_USE_IBV
        ("ibv", make_opt(opts.ibv_if), "Interface address for ibverbs")
        ("ibv-vector", make_opt(opts.ibv_comp_vector), "Interrupt vector (-1 for polled)")
        ("ibv-max-poll", make_opt(opts.ibv_max_poll), "Maximum number of times to poll in a row")
#endif
    ;

    hidden.add_options()
        ("source", po::value<std::vector<std::string>>()->composing(), "sources");
    all.add(desc);
    all.add(hidden);

    po::positional_options_description positional;
    positional.add("source", -1);
    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
            .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
            .options(all)
            .positional(positional)
            .run(), vm);
        po::notify(vm);
        if (vm.count("help"))
        {
            usage(std::cout, desc);
            std::exit(0);
        }
        if (!vm.count("source"))
            throw po::error("At least one port is required");
        opts.sources = vm["source"].as<std::vector<std::string>>();
#if SPEAD2_USE_NETMAP
        if (opts.sources.size() > 1 && opts.netmap_if != "")
        {
            throw po::error("--netmap cannot be used with multiple sources");
        }
#endif
        return opts;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        usage(std::cerr, desc);
        std::exit(2);
    }
}

void show_heap(const spead2::recv::heap &fheap, const options &opts)
{
    if (opts.quiet)
        return;
    time_point now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start;
    std::cout << std::showbase;
    std::cout << "Received heap " << fheap.get_cnt() << " at " << elapsed.count() << '\n';
    if (opts.descriptors)
    {
        std::vector<spead2::descriptor> descriptors = fheap.get_descriptors();
        for (const auto &descriptor : descriptors)
        {
            std::cout
                << "Descriptor for " << descriptor.name
                << " (" << std::hex << descriptor.id << ")\n"
                << "  description: " << descriptor.description << '\n'
                << "  format:      [";
            bool first = true;
            for (const auto &field : descriptor.format)
            {
                if (!first)
                    std::cout << ", ";
                first = false;
                std::cout << '(' << field.first << ", " << field.second << ')';
            }
            std::cout << "]\n";
            std::cout
                << "  dtype:       " << descriptor.numpy_header << '\n'
                << "  shape:       (";
            first = true;
            for (const auto &size : descriptor.shape)
            {
                if (!first)
                    std::cout << ", ";
                first = false;
                if (size == -1)
                    std::cout << "?";
                else
                    std::cout << size;
            }
            std::cout << ")\n";
        }
    }
    const auto &items = fheap.get_items();
    for (const auto &item : items)
    {
        std::cout << std::hex << item.id << std::dec
            << " = [" << item.length << " bytes]\n";
    }
    std::cout << std::noshowbase;
}

class callback_stream : public spead2::recv::stream
{
private:
    std::int64_t n_complete = 0;
    const options opts;

    virtual void heap_ready(spead2::recv::live_heap &&heap) override
    {
        if (heap.is_contiguous())
        {
            spead2::recv::heap frozen(std::move(heap));
            show_heap(frozen, opts);
            n_complete++;
        }
        else
            std::cout << "Discarding incomplete heap " << heap.get_cnt() << '\n';
    }

    std::promise<void> stop_promise;

public:
    template<typename... Args>
    callback_stream(const options &opts, Args&&... args)
        : spead2::recv::stream::stream(std::forward<Args>(args)...),
        opts(opts) {}

    virtual void stop_received() override
    {
        spead2::recv::stream::stop_received();
        stop_promise.set_value();
    }

    std::int64_t join()
    {
        std::future<void> future = stop_promise.get_future();
        future.get();
        return n_complete;
    }
};

template<typename It>
static std::unique_ptr<spead2::recv::stream> make_stream(
    spead2::thread_pool &thread_pool, const options &opts,
    It first_source, It last_source)
{
    using asio::ip::udp;

    std::unique_ptr<spead2::recv::stream> stream;
    spead2::bug_compat_mask bug_compat = opts.pyspead ? spead2::BUG_COMPAT_PYSPEAD_0_5_2 : 0;
    if (opts.ring)
        stream.reset(new spead2::recv::ring_stream<>(thread_pool, bug_compat, opts.heaps, opts.ring_heaps));
    else
        stream.reset(new callback_stream(opts, thread_pool, bug_compat, opts.heaps));

    if (opts.mem_pool)
    {
        std::shared_ptr<spead2::memory_pool> pool = std::make_shared<spead2::memory_pool>(
            opts.mem_lower, opts.mem_upper, opts.mem_max_free, opts.mem_initial);
        stream->set_memory_allocator(pool);
    }
    if (opts.memcpy_nt)
        stream->set_memcpy(spead2::MEMCPY_NONTEMPORAL);
    for (It i = first_source; i != last_source; ++i)
    {
        udp::resolver resolver(thread_pool.get_io_service());
        udp::resolver::query query(opts.bind, *i);
        udp::endpoint endpoint = *resolver.resolve(query);
#if SPEAD2_USE_NETMAP
        if (opts.netmap_if != "")
        {
            stream->emplace_reader<spead2::recv::netmap_udp_reader>(
                opts.netmap_if, endpoint.port());
        }
        else
#endif
#if SPEAD2_USE_IBV
        if (opts.ibv_if != "")
        {
            boost::asio::ip::address interface_address = boost::asio::ip::address::from_string(opts.ibv_if);
            stream->emplace_reader<spead2::recv::udp_ibv_reader>(
                endpoint, interface_address, opts.packet, opts.buffer,
                opts.ibv_comp_vector, opts.ibv_max_poll);
        }
        else
#endif
        {
            stream->emplace_reader<spead2::recv::udp_reader>(endpoint, opts.packet, opts.buffer);
        }
    }
    return stream;
}

int main(int argc, const char **argv)
{
    options opts = parse_args(argc, argv);

    spead2::thread_pool thread_pool(opts.threads);
    std::vector<std::unique_ptr<spead2::recv::stream> > streams;
    if (opts.joint)
    {
        streams.push_back(make_stream(thread_pool, opts, opts.sources.begin(), opts.sources.end()));
    }
    else
    {
        if (opts.sources.size() > 1 && opts.ring)
        {
            std::cerr << "Multiple streams cannot be used with --ring\n";
            std::exit(2);
        }
        for (auto it = opts.sources.begin(); it != opts.sources.end(); ++it)
            streams.push_back(make_stream(thread_pool, opts, it, it + 1));
    }

    std::int64_t n_complete = 0;
    if (opts.ring)
    {
        auto &stream = dynamic_cast<spead2::recv::ring_stream<> &>(*streams[0]);
        while (true)
        {
            try
            {
                spead2::recv::heap fh = stream.pop();
                n_complete++;
                show_heap(fh, opts);
            }
            catch (spead2::ringbuffer_stopped &e)
            {
                break;
            }
        }
    }
    else
    {
        for (const auto &ptr : streams)
        {
            auto &stream = dynamic_cast<callback_stream &>(*ptr);
            n_complete += stream.join();
        }
    }

    std::cout << "Received " << n_complete << " heaps\n";
    return 0;
}
