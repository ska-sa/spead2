/* Copyright 2015, 2018 SKA South Africa
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
#include <spead2/recv_tcp.h>
#if SPEAD2_USE_NETMAP
# include <spead2/recv_netmap.h>
#endif
#if SPEAD2_USE_IBV
# include <spead2/recv_udp_ibv.h>
#endif
#if SPEAD2_USE_PCAP
# include <spead2/recv_udp_pcap.h>
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
    bool tcp = false;
    std::string bind;
    std::size_t packet;
    std::size_t buffer;
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
    bool netmap = false;
#endif
#if SPEAD2_USE_IBV
    bool ibv = false;
    int ibv_comp_vector = 0;
    int ibv_max_poll = spead2::recv::udp_ibv_reader::default_max_poll;
#endif
    std::vector<std::string> sources;
};

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

static time_point start = std::chrono::high_resolution_clock::now();

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: spead2_recv [options] [host]:<port>|file\n";
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

template<typename T>
static po::typed_value<T> *make_opt_no_default(T &var)
{
    return po::value<T>(&var);
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
        ("tcp", make_opt(opts.tcp), "Receive data over TCP instead of UDP")
        ("bind", make_opt(opts.bind), "Interface address for multicast")
        ("packet", make_opt_no_default(opts.packet), "Maximum packet size to use")
        ("buffer", make_opt_no_default(opts.buffer), "Socket buffer size")
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
        ("netmap", make_opt(opts.netmap), "Use netmap")
#endif
#if SPEAD2_USE_IBV
        ("ibv", make_opt(opts.ibv), "Use ibverbs")
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
        if (!vm.count("packet"))
        {
            if (opts.tcp)
                opts.packet = spead2::recv::tcp_reader::default_max_size;
            else
                opts.packet = spead2::recv::udp_reader::default_max_size;
        }
        if (!vm.count("buffer"))
        {
            if (opts.tcp)
                opts.buffer = spead2::recv::tcp_reader::default_buffer_size;
            else
                opts.buffer = spead2::recv::udp_reader::default_buffer_size;
        }
#if SPEAD2_USE_IBV
        if (opts.ibv && opts.bind.empty())
            throw po::error("--ibv requires --bind");
        if (opts.tcp && opts.ibv)
            throw po::error("--ibv and --tcp are incompatible");
#endif
#if SPEAD2_USE_NETMAP
        if (opts.sources.size() > 1 && opts.netmap)
            throw po::error("--netmap cannot be used with multiple sources");
        if (opts.tcp && opts.netmap)
            throw po::error("--netmap and --tcp are incompatible");
#endif
#if SPEAD2_USE_IBV && SPEAD2_USE_NETMAP
        if (opts.ibv && opts.netmap)
            throw po::error("--ibv and --netmap are incompatible");
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
        else if (!opts.quiet)
            std::cout << "Discarding incomplete heap " << heap.get_cnt() << '\n';
    }

    std::promise<void> stop_promise;

public:
    template<typename... Args>
    callback_stream(const options &opts, Args&&... args)
        : spead2::recv::stream::stream(std::forward<Args>(args)...),
        opts(opts) {}

    ~callback_stream()
    {
        stop();
    }

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
    using asio::ip::tcp;

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
#if SPEAD2_USE_IBV
    std::vector<udp::endpoint> ibv_endpoints;
#endif
    for (It i = first_source; i != last_source; ++i)
    {
        std::string host = "";
        std::string port;
        auto colon = i->rfind(':');
        if (colon != std::string::npos)
        {
            host = i->substr(0, colon);
            port = i->substr(colon + 1);
        }
        else
            port = *i;

        bool is_pcap = false;
        try
        {
            boost::lexical_cast<std::uint16_t>(port);
        }
        catch (boost::bad_lexical_cast &)
        {
            is_pcap = true;
        }

        if (is_pcap)
        {
#if SPEAD2_USE_PCAP
            stream->emplace_reader<spead2::recv::udp_pcap_file_reader>(*i);
#else
            throw std::runtime_error("spead2 was compiled without pcap support");
#endif
        }
        else if (opts.tcp)
        {
            tcp::resolver resolver(thread_pool.get_io_service());
            tcp::resolver::query query(host, port, tcp::resolver::query::address_configured);
            tcp::endpoint endpoint = *resolver.resolve(query);
            stream->emplace_reader<spead2::recv::tcp_reader>(endpoint, opts.packet, opts.buffer);
        }
        else
        {
            udp::resolver resolver(thread_pool.get_io_service());
            udp::resolver::query query(host, port);
            udp::endpoint endpoint = *resolver.resolve(query);
#if SPEAD2_USE_NETMAP
            if (opts.netmap)
            {
                stream->emplace_reader<spead2::recv::netmap_udp_reader>(
                    opts.bind, endpoint.port());
            }
            else
#endif
#if SPEAD2_USE_IBV
            if (opts.ibv)
            {
                ibv_endpoints.push_back(endpoint);
            }
            else
#endif
            if (endpoint.address().is_multicast() && endpoint.address().is_v4()
                && !opts.bind.empty())
            {
                stream->emplace_reader<spead2::recv::udp_reader>(
                    endpoint, opts.packet, opts.buffer,
                    boost::asio::ip::address_v4::from_string(opts.bind));
            }
            else
            {
                if (!opts.bind.empty())
                    std::cerr << "--bind is only applicable to IPv4 multicast, ignoring\n";
                stream->emplace_reader<spead2::recv::udp_reader>(endpoint, opts.packet, opts.buffer);
            }
        }
    }
#if SPEAD2_USE_IBV
    if (!ibv_endpoints.empty())
    {
        boost::asio::ip::address interface_address = boost::asio::ip::address::from_string(opts.bind);
        stream->emplace_reader<spead2::recv::udp_ibv_reader>(
            ibv_endpoints, interface_address, opts.packet, opts.buffer,
            opts.ibv_comp_vector, opts.ibv_max_poll);
    }
#endif
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

    spead2::thread_pool stopper_thread_pool;
    boost::asio::signal_set signals(stopper_thread_pool.get_io_service());
    signals.add(SIGINT);
    signals.async_wait([&streams] (const boost::system::error_code &error, int signal_number) {
        if (!error)
            for (const std::unique_ptr<spead2::recv::stream> &stream : streams)
            {
                stream->stop();
            }
    });

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
    signals.cancel();
    spead2::recv::stream_stats stats;
    for (const auto &ptr : streams)
        stats += ptr->get_stats();

    std::cout << "Received " << n_complete << " heaps\n";
#define REPORT_STAT(field) (std::cout << #field ": " << stats.field << '\n')
    REPORT_STAT(heaps);
    REPORT_STAT(incomplete_heaps_evicted);
    REPORT_STAT(incomplete_heaps_flushed);
    REPORT_STAT(packets);
    REPORT_STAT(batches);
    REPORT_STAT(worker_blocked);
    REPORT_STAT(max_batch);
    REPORT_STAT(single_packet_heaps);
    REPORT_STAT(search_dist);
#undef REPORT_STAT
    return 0;
}
