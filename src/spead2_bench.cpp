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
#include <sstream>
#include <utility>
#include <cstdint>
#include <vector>
#include <deque>
#include <thread>
#include <memory>
#include <chrono>
#include <exception>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/format.hpp>
#include "common_thread_pool.h"
#include "common_defines.h"
#include "common_flavour.h"
#include "common_memory_pool.h"
#include "recv_udp.h"
#include "recv_heap.h"
#include "recv_live_heap.h"
#include "recv_ring_stream.h"
#include "send_heap.h"
#include "send_udp.h"
#include "send_stream.h"

namespace po = boost::program_options;
namespace asio = boost::asio;

struct options
{
    bool quiet = false;
    std::size_t packet_size = 9172;
    std::size_t heap_size = 4194304;
    int heap_address_bits = 40;
    std::size_t send_buffer = spead2::send::udp_stream::default_buffer_size;
    std::size_t recv_buffer = spead2::recv::udp_reader::default_buffer_size;
    std::size_t burst_size = spead2::send::stream_config::default_burst_size;
    std::size_t heaps = spead2::recv::stream_base::default_max_heaps;
    std::size_t mem_max_free = 12;
    std::size_t mem_initial = 8;

    std::string host;
    std::string port;
};

template<typename T>
static po::typed_value<T> *make_opt(T &var)
{
    return po::value<T>(&var)->default_value(var);
}

static void usage_master(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: spead2_bench master <host> <port> [options]\n";
    o << desc;
}

static options parse_master_args(int argc, const char **argv)
{
    options opts;
    po::options_description desc, hidden, all;
    desc.add_options()
        ("quiet", po::bool_switch(&opts.quiet)->default_value(opts.quiet), "Print only the final result")
        ("packet", make_opt(opts.packet_size), "Maximum packet size to use for UDP")
        ("heap-size", make_opt(opts.heap_size), "Payload size for heap")
        ("addr-bits", make_opt(opts.heap_address_bits), "Heap address bits")
        ("send-buffer", make_opt(opts.send_buffer), "Socket buffer size (sender)")
        ("recv-buffer", make_opt(opts.recv_buffer), "Socket buffer size (receiver)")
        ("burst", make_opt(opts.burst_size), "Send burst size")
        ("heaps", make_opt(opts.heaps), "Maximum number of in-flight heaps")
        ("mem-max-free", make_opt(opts.mem_max_free), "Maximum free memory buffers")
        ("mem-initial", make_opt(opts.mem_initial), "Initial free memory buffers")
        ("help,h", "Show help text");
    hidden.add_options()
        ("host", po::value<std::string>(&opts.host))
        ("port", po::value<std::string>(&opts.port));
    all.add(desc);
    all.add(hidden);

    po::positional_options_description positional;
    positional.add("host", 1);
    positional.add("port", 2);
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
            usage_master(std::cout, desc);
            std::exit(0);
        }
        return opts;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        usage_master(std::cerr, desc);
        std::exit(2);
    }
}

static bool measure_connection_once(
    const options &opts, double rate,
    std::int64_t num_heaps, std::int64_t required_heaps)
{
    using asio::ip::tcp;
    using asio::ip::udp;
    std::vector<std::uint8_t> data(opts.heap_size);

    /* Look up the address for the stream */
    asio::io_service io_service;
    udp::resolver resolver(io_service);
    udp::resolver::query query(opts.host, opts.port);
    udp::endpoint endpoint = *resolver.resolve(query);

    /* Initiate the control connection */
    tcp::iostream control(opts.host, opts.port);
    try
    {
        control.exceptions(std::ios::failbit);
        control << "start "
            << opts.packet_size << ' '
            << opts.heap_size << ' '
            << opts.heap_address_bits << ' '
            << opts.recv_buffer << ' '
            << opts.burst_size << ' '
            << opts.heaps << ' '
            << opts.mem_max_free << ' '
            << opts.mem_initial << std::endl;
        std::string response;
        std::getline(control, response);
        if (response != "ready")
        {
            std::cerr << "Did not receive 'ready' response\n";
            std::exit(1);
        }

        /* Construct the stream */
        spead2::thread_pool thread_pool;
        spead2::flavour flavour(4, 64, opts.heap_address_bits);
        /* Allow all heaps to be queued up at once. Since they all point to
         * the same payload, this should not cause excessive memory use.
         */
        spead2::send::stream_config config(
            opts.packet_size, rate, opts.burst_size, num_heaps + 1);
        spead2::send::udp_stream stream(
            thread_pool.get_io_service(), endpoint, config, opts.send_buffer);

        /* Build the heaps */
        std::vector<spead2::send::heap> heaps;
        for (std::int64_t i = 0; i < num_heaps; i++)
        {
            heaps.emplace_back(flavour);
            if (i + 1 < num_heaps)
                heaps.back().add_item(0x1234, data, false);
            else
                heaps.back().add_end();
        }

        /* Send the heaps */
        auto start = std::chrono::high_resolution_clock::now();
        std::int64_t transferred = 0;
        boost::system::error_code last_error;
        for (std::int64_t i = 0; i < num_heaps; i++)
        {
            auto callback = [&transferred, &last_error] (
                const boost::system::error_code &ec, spead2::item_pointer_t bytes)
            {
                if (!ec)
                    transferred += bytes;
                else
                    last_error = ec;
            };
            stream.async_send_heap(heaps[i], callback);
        }
        stream.flush();
        auto end = std::chrono::high_resolution_clock::now();
        if (last_error)
            throw boost::system::system_error(last_error);

        std::chrono::duration<double> elapsed_duration = end - start;
        double elapsed = elapsed_duration.count();
        double expected = transferred / rate;
        bool good = true;
        if (elapsed > 1.02 * expected)
        {
            if (!opts.quiet)
            {
                std::cout << boost::format("WARNING: transmission took longer than expected (%.3f > %.3f)\n")
                    % elapsed % expected;
            }
            good = false;
        }

        /* Get results */
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        control << "stop" << std::endl;
        std::int64_t received_heaps;
        control >> received_heaps;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        control.close();
        return good && received_heaps >= required_heaps;
    }
    catch (std::ios::failure &e)
    {
        std::cerr << "Connection error: " << control.error().message() << '\n';
        std::exit(1);
    }
    catch (boost::system::system_error &e)
    {
        std::cerr << "Transmission error: " << e.what() << '\n';
        std::exit(1);
    }
}

static bool measure_connection(
    const options &opts, double rate,
    std::int64_t num_heaps, std::int64_t required_heaps)
{
    int total = 0;
    for (int i = 0; i < 5; i++)
        total += measure_connection_once(opts, rate, num_heaps, required_heaps);
    return total >= 3;
}

static void main_master(int argc, const char **argv)
{
    options opts = parse_master_args(argc, argv);
    // These rates are in bytes
    double low = 0.0;
    double high = 5e9;
    while (high - low > 1e8 / 8)
    {
        // Need at least 1GB of data to overwhelm cache effects, and want at least
        // 1 second for warmup effects.
        double rate = (low + high) * 0.5;
        std::int64_t num_heaps = std::int64_t(std::max(1e9, rate) / opts.heap_size) + 2;
        bool good = measure_connection(opts, rate, num_heaps, num_heaps - 1);
        if (!opts.quiet)
            std::cout << boost::format("Rate: %.3f Gbps: %s\n") % (rate * 8e-9) % (good ? "GOOD" : "BAD");
        if (good)
            low = rate;
        else
            high = rate;
    }
    double rate = (low + high) * 0.5;
    double rate_gbps = rate * 8e-9;
    if (opts.quiet)
        std::cout << rate_gbps << '\n';
    else
        std::cout << boost::format("Sustainable rate: %.3f Gbps\n") % rate_gbps;
}

static void usage_slave(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: spead2_bench slave <port>\n";
    o << desc;
}

static options parse_slave_args(int argc, const char **argv)
{
    options opts;
    po::options_description desc, hidden, all;
    desc.add_options()
        ("help,h", "Show help text");
    hidden.add_options()
        ("port", po::value<std::string>(&opts.port));
    all.add(desc);
    all.add(hidden);

    po::positional_options_description positional;
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
            usage_slave(std::cout, desc);
            std::exit(0);
        }
        return opts;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        usage_slave(std::cerr, desc);
        std::exit(2);
    }
}

class slave_stream : public spead2::recv::stream
{
private:
    virtual void heap_ready(spead2::recv::live_heap &&live)
    {
        if (live.is_contiguous())
        {
            spead2::recv::heap heap(std::move(live));
            num_heaps++;
        }
    }

public:
    using spead2::recv::stream::stream;
    std::int64_t num_heaps = 0;
};

struct slave_connection
{
    spead2::thread_pool thread_pool;
    spead2::memory_pool memory_pool;
    slave_stream stream;

    slave_connection(const options &opts, const asio::ip::udp::endpoint &endpoint)
        : thread_pool(),
        memory_pool(
            opts.heap_size, opts.heap_size + 1024, opts.mem_max_free, opts.mem_initial),
        stream(thread_pool, 0, opts.heaps)
    {
        stream.emplace_reader<spead2::recv::udp_reader>(
            endpoint, opts.packet_size, opts.recv_buffer);
    }
};

static void main_slave(int argc, const char **argv)
{
    using asio::ip::tcp;
    using asio::ip::udp;
    options slave_opts = parse_slave_args(argc, argv);
    spead2::thread_pool thread_pool;

    /* Look up the bind address */
    udp::resolver resolver(thread_pool.get_io_service());
    udp::resolver::query query("0.0.0.0", slave_opts.port);
    udp::endpoint endpoint = *resolver.resolve(query);

    tcp::acceptor acceptor(thread_pool.get_io_service(),
                           tcp::endpoint(endpoint.address(), endpoint.port()));
    while (true)
    {
        tcp::iostream control;
        acceptor.accept(*control.rdbuf());
        std::unique_ptr<slave_connection> connection;
        while (true)
        {
            std::string line;
            std::getline(control, line);
            if (!control)
                break;
            std::istringstream toks(line);
            std::string cmd;
            toks >> cmd;
            if (cmd == "start")
            {
                if (connection)
                {
                    std::cerr <<  "Start received while already running\n";
                    continue;
                }
                options opts;
                toks >> opts.packet_size
                    >> opts.heap_size
                    >> opts.heap_address_bits
                    >> opts.recv_buffer
                    >> opts.burst_size
                    >> opts.heaps
                    >> opts.mem_max_free
                    >> opts.mem_initial;
                connection.reset(new slave_connection(opts, endpoint));
                control << "ready" << std::endl;
            }
            else if (cmd == "stop")
            {
                if (!connection)
                {
                    std::cerr << "Stop received when already stopped\n";
                    continue;
                }
                connection->stream.stop();
                auto num_heaps = connection->stream.num_heaps;
                connection.reset();
                std::cerr << num_heaps << '\n';
                control << num_heaps << std::endl;
            }
            else if (cmd == "exit")
                break;
            else
                std::cerr << "Bad command: " << line << '\n';
        }
    }
}

int main(int argc, const char **argv)
{
    if (argc >= 2 && argv[1] == std::string("master"))
        main_master(argc - 1, argv + 1);
    else if (argc >= 2 && argv[1] == std::string("slave"))
        main_slave(argc - 1, argv + 1);
    else
    {
        std::cerr << "Usage:\n"
            << "    spead2_bench master <host> <port> [options]\n"
            << "OR\n"
            << "    spead2_bench slave <port> [options]\n";
        return 2;
    }

    return 0;
}
