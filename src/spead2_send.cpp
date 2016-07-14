/* Copyright 2016 SKA South Africa
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
#include <vector>
#include <string>
#include <deque>
#include <utility>
#include <sstream>
#include <exception>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/send_stream.h>
#include <spead2/send_udp.h>
#if SPEAD2_USE_IBV
# include <spead2/send_udp_ibv.h>
#endif

namespace po = boost::program_options;
namespace asio = boost::asio;
using boost::asio::ip::udp;

struct options
{
    std::size_t heap_size = 4194304;
    std::size_t items = 1;
    std::int64_t heaps = -1;
    bool pyspead = false;
    int addr_bits = 40;
    std::size_t packet = spead2::send::stream_config::default_max_packet_size;
    std::size_t buffer = spead2::send::udp_stream::default_buffer_size;
    std::size_t burst = spead2::send::stream_config::default_burst_size;
    int threads = 1;
    double rate = 0.0;
#if SPEAD2_USE_IBV
    std::string ibv_if;
    int ibv_comp_vector = 0;
    int ibv_max_poll = spead2::send::udp_ibv_stream::default_max_poll;
#endif
    std::string host;
    std::string port;
};

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: spead2_send [options] <host> <port>\n";
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
static po::typed_value<T> *make_required_opt(T &var)
{
    return po::value<T>(&var);
}

static options parse_args(int argc, const char **argv)
{
    options opts;
    po::options_description desc, hidden, all;
    desc.add_options()
        ("heap-size", make_opt(opts.heap_size), "Payload size for heap")
        ("items", make_opt(opts.items), "Number of items per heap")
        ("heaps", make_opt(opts.heaps), "Number of data heaps to send (-1=infinite)")
        ("pyspead", make_opt(opts.pyspead), "Be bug-compatible with PySPEAD")
        ("addr-bits", make_opt(opts.addr_bits), "Heap address bits")
        ("packet", make_opt(opts.packet), "Maximum packet size to send")
        ("buffer", make_opt(opts.buffer), "Socket buffer size")
        ("burst", make_opt(opts.burst), "Burst size")
        ("threads", make_opt(opts.threads), "Number of worker threads")
        ("rate", make_opt(opts.rate), "Transmission rate bound (Gb/s)")
#if SPEAD2_USE_IBV
        ("ibv", make_opt(opts.ibv_if), "Interface address for ibverbs")
        ("ibv-vector", make_opt(opts.ibv_comp_vector), "Interrupt vector (-1 for polled)")
        ("ibv-max-poll", make_opt(opts.ibv_max_poll), "Maximum number of times to poll in a row")
#endif
    ;
    hidden.add_options()
        ("host", make_required_opt(opts.host), "Destination host")
        ("port", make_required_opt(opts.port), "Destination port")
    ;
    all.add(desc);
    all.add(hidden);

    po::positional_options_description positional;
    positional.add("host", 1);
    positional.add("port", 1);
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
        if (!vm.count("host") || !vm.count("port"))
            throw po::error("too few positional options have been specified on the command line");
        return opts;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        usage(std::cerr, desc);
        std::exit(2);
    }
}

// Sends a heap, returning a future instead of using a completion handler
std::future<std::size_t> async_send_heap(spead2::send::stream &stream, const spead2::send::heap &heap)
{
    auto promise = std::make_shared<std::promise<std::size_t>>();
    auto handler = [promise] (boost::system::error_code ec, std::size_t bytes_transferred)
    {
        if (ec)
            promise->set_exception(std::make_exception_ptr(boost::system::system_error(ec)));
        else
            promise->set_value(bytes_transferred);
    };
    stream.async_send_heap(heap, handler);
    return promise->get_future();
}

int run(spead2::send::stream &stream, const options &opts)
{
    typedef std::pair<float, float> item_t;
    std::size_t elements = opts.heap_size / (opts.items * sizeof(item_t));
    std::size_t heap_size = elements * opts.items * sizeof(item_t);
    if (heap_size != opts.heap_size)
    {
        std::cerr << "Heap size is not an exact multiple: using " << heap_size << " instead of " << opts.heap_size << '\n';
    }

    spead2::flavour f(spead2::maximum_version, 64, opts.addr_bits,
                      opts.pyspead ? spead2::BUG_COMPAT_PYSPEAD_0_5_2 : 0);

    std::vector<spead2::descriptor> descriptors;
    std::vector<std::unique_ptr<item_t[]>> values;
    descriptors.reserve(opts.items);
    values.reserve(opts.items);
    for (std::size_t i = 0; i < opts.items; i++)
    {
        spead2::descriptor d;
        d.id = 0x1000 + i;
        std::ostringstream sstr;
        sstr << "Test item " << i;
        d.name = sstr.str();
        d.description = "A test item with arbitrary value";
        sstr.str("");
        sstr << "{'shape': (" << elements << ",), 'fortran_order': False, 'descr': '<c8'}";
        d.numpy_header = sstr.str();
        descriptors.push_back(d);
        std::unique_ptr<item_t[]> item(new item_t[elements]);
        values.push_back(std::move(item));
    }

    std::deque<std::future<std::size_t>> futures;
    std::deque<spead2::send::heap> heaps;
    std::uint64_t sent = 0;
    while (opts.heaps < 0 || sent <= std::uint64_t(opts.heaps))
    {
        if (futures.size() >= 2)
        {
            futures.front().get();
            futures.pop_front();
            heaps.pop_front();
        }

        heaps.emplace_back(f);
        spead2::send::heap &heap = heaps.back();
        if (sent == 0)
        {
            for (const auto &d : descriptors)
                heap.add_descriptor(d);
        }
        if (opts.heaps >= 0 && sent == std::uint64_t(opts.heaps))
        {
            heap.add_end();
        }
        else
        {
            for (std::size_t i = 0; i < opts.items; i++)
                heap.add_item(0x1000 + i, values[i].get(), elements * sizeof(item_t), true);
        }
        futures.push_back(async_send_heap(stream, heap));
        sent++;
    }
    while (futures.size() > 0)
    {
        futures.front().get();
        futures.pop_front();
        heaps.pop_front();
    }
    return 0;
}

int main(int argc, const char **argv)
{
    options opts = parse_args(argc, argv);

    spead2::thread_pool thread_pool(opts.threads);
    udp::resolver resolver(thread_pool.get_io_service());
    udp::resolver::query query(opts.host, opts.port);
    auto it = resolver.resolve(query);
    spead2::send::stream_config config(
        opts.packet, opts.rate * 1024 * 1024 * 1024 / 8, opts.burst);
    std::unique_ptr<spead2::send::stream> stream;
#if SPEAD2_USE_IBV
    if (opts.ibv_if != "")
    {
        boost::asio::ip::address interface_address = boost::asio::ip::address::from_string(opts.ibv_if);
        stream.reset(new spead2::send::udp_ibv_stream(
                thread_pool.get_io_service(), *it, config,
                interface_address, opts.buffer, 1,
                opts.ibv_comp_vector, opts.ibv_max_poll));
    }
    else
#endif
    {
        stream.reset(new spead2::send::udp_stream(
                thread_pool.get_io_service(), *it, config, opts.buffer));
    }
    return run(*stream, opts);
}
