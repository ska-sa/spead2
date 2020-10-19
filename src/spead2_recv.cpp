/* Copyright 2015, 2018-2020 National Research Foundation (SARAO)
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
#include <random>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/common_endian.h>
#include <spead2/recv_udp.h>
#include <spead2/recv_tcp.h>
#if SPEAD2_USE_IBV
# include <spead2/recv_udp_ibv.h>
#endif
#if SPEAD2_USE_PCAP
# include <spead2/recv_udp_pcap.h>
#endif
#include <spead2/recv_heap.h>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_ring_stream.h>
#include "spead2_cmdline.h"

namespace po = boost::program_options;
namespace asio = boost::asio;

struct options
{
    bool quiet = false;
    bool descriptors = false;
    bool joint = false;
    int threads = 1;
    bool verify = false;
    std::vector<std::string> sources;
    spead2::protocol_options protocol;
    spead2::recv::receiver_options receiver;
};

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

static time_point start = std::chrono::high_resolution_clock::now();

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: spead2_recv [options] [host]:<port>|file\n";
    o << desc;
}

static options parse_args(int argc, const char **argv)
{
    options opts;
    po::options_description desc, hidden, all;
    spead2::option_adder adder(desc);
    opts.protocol.enumerate(adder);
    opts.receiver.enumerate(adder);
    desc.add_options()
        ("quiet", spead2::make_value_semantic(&opts.quiet), "Only show total of heaps received")
        ("descriptors", spead2::make_value_semantic(&opts.descriptors), "Show descriptors")
        ("joint", spead2::make_value_semantic(&opts.joint), "Treat all sources as a single stream")
        ("threads", spead2::make_value_semantic(&opts.threads), "Number of worker threads")
        ("verify", spead2::make_value_semantic(&opts.verify), "Verify payload (use spead2_send with same option")
    ;

    hidden.add_options()
        ("source", spead2::make_value_semantic(&opts.sources), "sources");
    desc.add_options()
        ("help,h", "Show help text");
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
        if (opts.sources.empty())
            throw po::error("At least one source is required");
        opts.protocol.notify();
        opts.receiver.notify(opts.protocol);
        return opts;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        usage(std::cerr, desc);
        std::exit(2);
    }
}

static void show_heap(const spead2::recv::heap &fheap, const options &opts)
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
            << " = [" << item.length << " bytes]";
        if (item.is_immediate)
            std::cout << " = " << std::hex << item.immediate_value;
        std::cout << '\n';
    }
    std::cout << std::noshowbase;
}

/* O(log(z)) version of engine.discard(z), which in GCC seems to be implemented in O(z). */
template<std::uint_fast32_t a, std::uint_fast32_t m>
static void fast_discard(std::linear_congruential_engine<std::uint_fast32_t, a, 0, m> &engine,
                         unsigned long long z)
{
    if (z == 0)
        return;
    // There is no way to directly observe the current state. We can only
    // advance to the following state.
    std::uint64_t x = engine();
    z--;
    // Multiply by a^z mod m
    std::uint64_t apow = a;
    while (z > 0)
    {
        if (z & 1)
            x = x * apow % m;
        apow = apow * apow % m;
        z >>= 1;
    }
    engine.seed(x);
}

static void verify_heap(const spead2::recv::heap &fheap, const options &opts)
{
    if (!opts.verify)
        return;
    const auto &items = fheap.get_items();

    typedef uint32_t element_t;
    bool first = true;
    std::size_t elements = 0;
    std::minstd_rand generator;
    for (const auto &item : items)
    {
        if (item.id < 0x1000)
            continue;
        if (first)
        {
            elements = item.length / sizeof(element_t);
            // The first heap gets numbered 1 rather than 0
            std::uint64_t start_pos = elements * items.size() * (fheap.get_cnt() - 1);
            fast_discard(generator, start_pos);
            first = false;
        }
        if (item.length != elements * sizeof(element_t))
        {
            std::cerr << "Heap " << fheap.get_cnt()
                << ", item 0x" << std::hex << item.id << std::dec
                << " has an inconsistent length\n";
            std::exit(1);
        }
        const element_t *data = reinterpret_cast<const element_t *>(item.ptr);
        for (std::size_t i = 0; i < elements; i++)
        {
            element_t expected = generator();
            element_t actual = spead2::betoh(data[i]);
            if (expected != actual)
            {
                std::cerr << "Verification mismatch in heap " << fheap.get_cnt()
                    << ", item 0x" << std::hex << item.id << std::dec
                    << " offset " << i
                    << "\nexpected 0x" << std::hex << expected << ", actual 0x" << actual << std::dec
                    << std::endl;
                std::exit(1);
            }
        }
    }

    if (first && !fheap.is_end_of_stream())
    {
        spead2::log_warning("Heap %d has no verifiable items but is not an end-of-stream heap",
                            fheap.get_cnt());
    }
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
            verify_heap(frozen, opts);
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
    spead2::recv::stream_config config = opts.receiver.make_stream_config(opts.protocol);

    if (opts.receiver.ring)
    {
        spead2::recv::ring_stream_config ring_config = opts.receiver.make_ring_stream_config();
        stream.reset(new spead2::recv::ring_stream<>(thread_pool, config, ring_config));
    }
    else
        stream.reset(new callback_stream(opts, thread_pool, config));

    std::vector<std::string> endpoints(first_source, last_source);
    opts.receiver.add_readers(*stream, endpoints, opts.protocol, true);
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
        if (opts.sources.size() > 1 && opts.receiver.ring)
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
    if (opts.receiver.ring)
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
    for (auto &ptr : streams)
    {
        /* Even though we've seen the stop condition, if we don't explicitly
         * stop the stream then a race condition means we might not see the
         * last batch of statistics updates.
         */
        ptr->stop();
        stats += ptr->get_stats();
    }

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
