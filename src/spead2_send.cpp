/* Copyright 2016, 2017, 2019 SKA South Africa
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
#include <spead2/common_semaphore.h>
#include <spead2/send_stream.h>
#include <spead2/send_udp.h>
#include <spead2/send_tcp.h>
#include <spead2/common_features.h>
#if SPEAD2_USE_IBV
# include <spead2/send_udp_ibv.h>
#endif

namespace po = boost::program_options;
namespace asio = boost::asio;
using boost::asio::ip::udp;
using boost::asio::ip::tcp;

struct options
{
    std::size_t heap_size = 4194304;
    std::size_t items = 1;
    std::int64_t heaps = -1;
    bool pyspead = false;
    bool tcp = false;
    std::string bind;
    int addr_bits = 40;
    std::size_t packet = spead2::send::stream_config::default_max_packet_size;
    std::size_t buffer;
    std::size_t burst = spead2::send::stream_config::default_burst_size;
    double burst_rate_ratio = spead2::send::stream_config::default_burst_rate_ratio;
    std::size_t max_heaps = spead2::send::stream_config::default_max_heaps;
    double rate = 0.0;
    int ttl = 1;
#if SPEAD2_USE_IBV
    bool ibv = false;
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
static po::typed_value<T> *make_opt_no_default(T &var)
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
        ("tcp", make_opt(opts.tcp), "Use TCP instead than UDP")
        ("bind", make_opt(opts.bind), "Local address to bind sockets to")
        ("packet", make_opt(opts.packet), "Maximum packet size to send")
        ("buffer", make_opt_no_default(opts.buffer), "Socket buffer size")
        ("burst", make_opt(opts.burst), "Burst size")
        ("burst-rate-ratio", make_opt(opts.burst_rate_ratio), "Hard rate limit, relative to --rate")
        ("max-heaps", make_opt(opts.max_heaps), "Maximum heaps in flight")
        ("rate", make_opt(opts.rate), "Transmission rate bound (Gb/s)")
        ("ttl", make_opt(opts.ttl), "TTL for multicast target")
#if SPEAD2_USE_IBV
        ("ibv", make_opt(opts.ibv), "Use ibverbs")
        ("ibv-vector", make_opt(opts.ibv_comp_vector), "Interrupt vector (-1 for polled)")
        ("ibv-max-poll", make_opt(opts.ibv_max_poll), "Maximum number of times to poll in a row")
#endif
    ;
    hidden.add_options()
        ("host", make_opt_no_default(opts.host), "Destination host")
        ("port", make_opt_no_default(opts.port), "Destination port")
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
        if (!vm.count("buffer"))
        {
            if (opts.tcp)
                opts.buffer = spead2::send::tcp_stream::default_buffer_size;
            else
                opts.buffer = spead2::send::udp_stream::default_buffer_size;
        }
#if SPEAD2_USE_IBV
        if (opts.ibv && opts.bind.empty())
            throw po::error("--ibv requires --bind");
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

namespace
{

class sender
{
private:
    spead2::send::stream &stream;
    const std::size_t max_heaps;
    const std::int64_t n_heaps;
    const spead2::flavour flavour;

    spead2::send::heap first_heap;                           // has descriptors
    std::vector<spead2::send::heap> heaps;
    spead2::send::heap last_heap;                            // has end-of-stream marker
    typedef std::pair<float, float> item_t;
    std::vector<std::unique_ptr<item_t[]>> values;

    std::uint64_t bytes_transferred = 0;
    boost::system::error_code error;
    spead2::semaphore done_sem{0};

    const spead2::send::heap &get_heap(std::uint64_t idx) const noexcept;

    void callback(std::uint64_t idx, const boost::system::error_code &ec, std::size_t bytes_transferred);

public:
    sender(spead2::send::stream &stream, const options &opts);
    std::uint64_t run();
};

sender::sender(spead2::send::stream &stream, const options &opts)
    : stream(stream),
    max_heaps((opts.heaps < 0 || std::uint64_t(opts.heaps) >= opts.max_heaps)
              ? opts.max_heaps : opts.heaps + 1),
    n_heaps(opts.heaps),
    flavour(spead2::maximum_version, 64, opts.addr_bits,
            opts.pyspead ? spead2::BUG_COMPAT_PYSPEAD_0_5_2 : 0),
    first_heap(flavour),
    last_heap(flavour)
{
    heaps.reserve(max_heaps);
    for (std::size_t i = 0; i < max_heaps; i++)
        heaps.emplace_back(flavour);

    const std::size_t elements = opts.heap_size / (opts.items * sizeof(item_t));
    const std::size_t heap_size = elements * opts.items * sizeof(item_t);
    if (heap_size != opts.heap_size)
    {
        std::cerr << "Heap size is not an exact multiple: using " << heap_size << " instead of " << opts.heap_size << '\n';
    }

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
        first_heap.add_descriptor(d);
        std::unique_ptr<item_t[]> item(new item_t[elements]);
        item_t *ptr = item.get();
        values.push_back(std::move(item));
        for (std::size_t j = 0; j < max_heaps; j++)
            heaps[j].add_item(0x1000 + i, ptr, elements * sizeof(item_t), true);
        first_heap.add_item(0x1000 + i, ptr, elements * sizeof(item_t), true);
    }
    last_heap.add_end();
}

const spead2::send::heap &sender::get_heap(std::uint64_t idx) const noexcept
{
    if (idx == 0)
        return first_heap;
    else if (n_heaps >= 0 && idx == std::uint64_t(n_heaps))
        return last_heap;
    else
        return heaps[idx % max_heaps];
}

void sender::callback(std::uint64_t idx, const boost::system::error_code &ec, std::size_t bytes_transferred)
{
    this->bytes_transferred += bytes_transferred;
    if (ec && !error)
        error = ec;
    if (error)
    {
        done_sem.put();
        return;
    }

    if (n_heaps == -1 || std::uint64_t(n_heaps) - idx >= max_heaps)
    {
        idx += max_heaps;
        stream.async_send_heap(get_heap(idx), [this, idx] (const boost::system::error_code &ec, std::size_t bytes_transferred) {
            callback(idx, ec, bytes_transferred); });
    }
    else
        done_sem.put();
}

std::uint64_t sender::run()
{
    bytes_transferred = 0;
    error = boost::system::error_code();
    /* Send the initial heaps from the worker thread. This ensures that no
     * callbacks can happen until the initial heaps are all sent, which would
     * otherwise lead to heaps being queued out of order. For this benchmark it
     * doesn't really matter since the heaps are all the same, but it makes it
     * a more realistic benchmark.
     */
    stream.get_io_service().post([this] {
        for (int i = 0; i < max_heaps; i++)
            stream.async_send_heap(get_heap(i), [this, i] (const boost::system::error_code &ec, std::size_t bytes_transferred) {
                callback(i, ec, bytes_transferred); });
    });
    for (int i = 0; i < max_heaps; i++)
        semaphore_get(done_sem);
    if (error)
        throw boost::system::system_error(error);
    return bytes_transferred;
}

} // anonymous namespace

static int run(spead2::send::stream &stream, const options &opts)
{
    sender s(stream, opts);

    auto start_time = std::chrono::high_resolution_clock::now();
    std::uint64_t sent_bytes = s.run();
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop_time - start_time;
    double elapsed_s = elapsed.count();
    std::cout
        << "Sent " << sent_bytes << " bytes in " << elapsed_s << " seconds, "
        << sent_bytes * 8.0e-9 / elapsed_s << " Gb/s\n";
    return 0;
}

template <typename Proto>
static boost::asio::ip::basic_endpoint<Proto> get_endpoint(
    boost::asio::io_service &io_service, const options &opts)
{
    typedef boost::asio::ip::basic_resolver<Proto> resolver_type;
    resolver_type resolver(io_service);
    typename resolver_type::query query(opts.host, opts.port);
    return *resolver.resolve(query);
}

int main(int argc, const char **argv)
{
    options opts = parse_args(argc, argv);

    spead2::thread_pool thread_pool(1);
    spead2::send::stream_config config(
        opts.packet, opts.rate * 1000 * 1000 * 1000 / 8, opts.burst,
        opts.max_heaps, opts.burst_rate_ratio);
    std::unique_ptr<spead2::send::stream> stream;
    auto &io_service = thread_pool.get_io_service();
    boost::asio::ip::address interface_address;
    if (!opts.bind.empty())
        interface_address = boost::asio::ip::address::from_string(opts.bind);

    if (opts.tcp) {
        tcp::endpoint endpoint = get_endpoint<tcp>(io_service, opts);
        auto promise = std::promise<void>();
        auto connect_handler = [&promise] (const boost::system::error_code &e) {
            if (e)
                promise.set_exception(std::make_exception_ptr(boost::system::system_error(e)));
            else
                promise.set_value();
        };
        stream.reset(new spead2::send::tcp_stream(
                    io_service, connect_handler, endpoint, config, opts.buffer, interface_address));
        promise.get_future().get();
    }
    else
    {
        udp::endpoint endpoint = get_endpoint<udp>(io_service, opts);
#if SPEAD2_USE_IBV
        if (opts.ibv)
        {
            stream.reset(new spead2::send::udp_ibv_stream(
                    io_service, endpoint, config,
                    interface_address, opts.buffer, opts.ttl,
                    opts.ibv_comp_vector, opts.ibv_max_poll));
        }
        else
#endif
        {
            if (endpoint.address().is_multicast())
            {
                if (endpoint.address().is_v4())
                    stream.reset(new spead2::send::udp_stream(
                            io_service, endpoint, config, opts.buffer,
                            opts.ttl, interface_address));
                else
                {
                    if (!opts.bind.empty())
                        std::cerr << "--bind is not yet supported for IPv6 multicast, ignoring\n";
                    stream.reset(new spead2::send::udp_stream(
                            io_service, endpoint, config, opts.buffer,
                            opts.ttl));
                }
            }
            else
            {
                stream.reset(new spead2::send::udp_stream(
                        io_service, endpoint, config, opts.buffer, interface_address));
            }
        }
    }
    return run(*stream, opts);
}
