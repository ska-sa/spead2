/* Copyright 2015, 2017, 2019-2020 National Research Foundation (SARAO)
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
#include <iomanip>
#include <locale>
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
#include <spead2/common_thread_pool.h>
#include <spead2/common_defines.h>
#include <spead2/common_flavour.h>
#include <spead2/common_memory_pool.h>
#include <spead2/common_semaphore.h>
#include <spead2/recv_udp.h>
#include <spead2/recv_heap.h>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_ring_stream.h>
#include <spead2/recv_mem.h>
#include <spead2/send_heap.h>
#include <spead2/send_udp.h>
#include <spead2/send_stream.h>
#include <spead2/send_streambuf.h>
#if SPEAD2_USE_IBV
# include <spead2/send_udp_ibv.h>
# include <spead2/recv_udp_ibv.h>
#endif
#include "spead2_cmdline.h"

namespace po = boost::program_options;
namespace asio = boost::asio;

class option_writer
{
private:
    std::ostream &out;

public:
    option_writer(std::ostream &o) : out(o) {}

    template<typename T>
    void operator()(const std::string &name, const std::string &description, const T *value) const
    {
        out << name << " = " << boost::lexical_cast<std::string>(*value) << '\n';
    }

    template<typename T>
    void operator()(const std::string &name, const std::string &description, const boost::optional<T> *value) const
    {
        if (*value)
            out << name << " = " << boost::lexical_cast<std::string>(**value) << '\n';
    }
};

struct options
{
    bool quiet = false;
    std::size_t heap_size = 4194304;
    std::string multicast;
    std::string endpoint;    // host:port for master, port for agent

    spead2::protocol_options protocol;
    spead2::recv::receiver_options receiver;
    spead2::send::sender_options sender;

    // Only used for the protocol used by the master to communicate the options
    // to the agent.
    template<typename T>
    void enumerate_wire(const T &callback)
    {
        callback("multicast", "", &multicast);
        protocol.enumerate(callback);
        receiver.enumerate(callback);
    }
};

enum class command_mode
{
    MASTER,
    AGENT,
    MEM
};

static void usage(std::ostream &o, const po::options_description &desc, command_mode mode)
{
    switch (mode)
    {
    case command_mode::MASTER:
        o << "Usage: spead2_bench master <host> <port> [options]\n";
        break;
    case command_mode::AGENT:
        o << "Usage: spead2_bench agent <port>\n";
        break;
    case command_mode::MEM:
        o << "Usage spead2_bench mem [options]\n";
        break;
    }
    o << desc;
}

static options parse_args(int argc, const char **argv, command_mode mode)
{
    options opts;
    po::options_description desc, hidden, all;
    po::positional_options_description positional;

    std::map<std::string, std::string> protocol_map, receiver_map, sender_map;
    // Empty values suppress options that aren't applicable
    // Memory pool sizes are managed automatically
    protocol_map["tcp"] = "";
    receiver_map["mem-pool"] = "";
    receiver_map["mem-lower"] = "";
    receiver_map["mem-upper"] = "";
    switch (mode)
    {
    case command_mode::MEM:
        receiver_map["buffer"] = "";
        receiver_map["ibv"] = "";
        receiver_map["ibv-vector"] = "";
        receiver_map["ibv-max-poll"] = "";
        receiver_map["bind"] = "";
        break;
    case command_mode::MASTER:
        receiver_map["bind"] = "recv-bind";
        receiver_map["buffer"] = "recv-buffer";
        receiver_map["ibv"] = "recv-ibv";
        receiver_map["ibv-vector"] = "recv-ibv-vector";
        receiver_map["ibv-max-poll"] = "recv-ibv-max-poll";
        sender_map["bind"] = "send-bind";
        sender_map["buffer"] = "send-buffer";
        sender_map["packet"] = "";  // Packet size is taken from receiver
        sender_map["ibv"] = "send-ibv";
        sender_map["ibv-vector"] = "send-ibv-vector";
        sender_map["ibv-max-poll"] = "send-max-poll";
        sender_map["rate"] = "";   // Controlled by test
        break;
    case command_mode::AGENT:
        break;
    }

    if (mode == command_mode::MASTER || mode == command_mode::MEM)
    {
        desc.add_options()
            ("quiet", spead2::make_value_semantic(&opts.quiet), "Print only the final result")
            ("heap-size", spead2::make_value_semantic(&opts.heap_size), "Payload size for heap");
        opts.protocol.enumerate(spead2::option_adder(desc, protocol_map));
        opts.receiver.enumerate(spead2::option_adder(desc, receiver_map));
    }
    if (mode == command_mode::MASTER)
    {
        opts.sender.enumerate(spead2::option_adder(desc, sender_map));
        desc.add_options()
            ("multicast", po::value<std::string>(&opts.multicast), "Multicast group to use, instead of unicast")
        ;
        hidden.add_options()
            ("endpoint", po::value<std::string>(&opts.endpoint));
        positional.add("endpoint", 1);
    }
    else if (mode == command_mode::AGENT)
    {
        hidden.add_options()
            ("port", po::value<std::string>(&opts.endpoint));
        positional.add("port", 1);
    }
    desc.add_options()
        ("help,h", "Show help text");
    all.add(desc);
    all.add(hidden);

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
            usage(std::cout, desc, mode);
            std::exit(0);
        }
        if ((mode == command_mode::MASTER && !vm.count("endpoint"))
             || (mode == command_mode::AGENT && !vm.count("port")))
        {
            throw po::error("too few positional options have been specified on the command line");
        }
#if SPEAD2_USE_IBV
        if (opts.sender.ibv && opts.multicast.empty())
        {
            throw po::error("--send-ibv requires --multicast");
        }
#endif
        if (mode != command_mode::AGENT)
        {
            // Initialise memory pool sizes based on heap size
            opts.receiver.mem_pool = true;
            opts.receiver.mem_lower = opts.heap_size;
            opts.receiver.mem_upper = opts.heap_size + 1024;  // more than enough for overheads
            opts.protocol.notify();
            opts.receiver.notify(opts.protocol);
            opts.sender.max_packet_size = *opts.receiver.max_packet_size;
            opts.sender.notify(opts.protocol);
        }
        return opts;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        usage(std::cerr, desc, mode);
        std::exit(2);
    }
}

namespace
{

class sender
{
private:
    spead2::send::stream &stream;
    const std::vector<spead2::send::heap> &heaps;
    const std::size_t max_heaps;

    std::uint64_t bytes_transferred = 0;
    boost::system::error_code error;
    spead2::semaphore done_sem{0};

    void callback(std::size_t idx, const boost::system::error_code &ec, std::size_t bytes_transferred);

public:
    sender(spead2::send::stream &stream, const std::vector<spead2::send::heap> &heaps,
           const options &opts);
    std::int64_t run();
};

sender::sender(spead2::send::stream &stream, const std::vector<spead2::send::heap> &heaps,
               const options &opts)
    : stream(stream), heaps(heaps), max_heaps(std::min(opts.sender.max_heaps, heaps.size()))
{
}

void sender::callback(std::size_t idx, const boost::system::error_code &ec, std::size_t bytes_transferred)
{
    this->bytes_transferred += bytes_transferred;
    if (ec && !error)
        error = ec;
    if (!error && idx + max_heaps < heaps.size())
    {
        idx += max_heaps;
        stream.async_send_heap(heaps[idx], [this, idx] (const boost::system::error_code &ec, std::size_t bytes_transferred) {
            callback(idx, ec, bytes_transferred); });
    }
    else
        done_sem.put();
}

std::int64_t sender::run()
{
    bytes_transferred = 0;
    error = boost::system::error_code();
    /* See comments in spead2_send.cpp for the explanation of why this is
     * posted rather than run directly.
     */
    stream.get_io_service().post([this] {
        for (int i = 0; i < max_heaps; i++)
            stream.async_send_heap(heaps[i], [this, i] (const boost::system::error_code &ec, std::size_t bytes_transferred) {
                callback(i, ec, bytes_transferred); });
    });
    for (int i = 0; i < max_heaps; i++)
        semaphore_get(done_sem);
    if (error)
        throw boost::system::system_error(error);
    return bytes_transferred;
}

} // anonymous namespace

static std::pair<bool, double> measure_connection_once(
    const options &opts, double rate,
    std::int64_t num_heaps, std::int64_t required_heaps)
{
    using asio::ip::tcp;
    using asio::ip::udp;
    std::vector<std::uint8_t> data(opts.heap_size);

    /* Look up the address for the stream */
    std::string endpoint = opts.multicast.empty() ? opts.endpoint : opts.multicast;

    /* Initiate the control connection */
    tcp::endpoint control_endpoint = spead2::parse_endpoint<tcp>(opts.endpoint, false);
    // iostream constructor wants to do a query, so we have to turn everything
    // into strings for it.
    tcp::iostream control(
        control_endpoint.protocol(),
        control_endpoint.address().to_string(),
        boost::lexical_cast<std::string>(control_endpoint.port()),
        boost::asio::ip::resolver_query_base::numeric_host
        | boost::asio::ip::resolver_query_base::numeric_service);
    try
    {
        control.exceptions(std::ios::failbit);
        control << "start\n";
        option_writer writer(control);
        const_cast<options &>(opts).enumerate_wire(writer);
        control << "end_config\n";

        std::string response;
        std::getline(control, response);
        if (response != "ready")
        {
            std::cerr << "Did not receive 'ready' response\n";
            std::exit(1);
        }

        /* Construct the stream */
        spead2::send::sender_options sender_options = opts.sender;
        // sender_options expects rate in Gb/s
        sender_options.rate = rate * 8e-9;

        spead2::thread_pool thread_pool;
        spead2::flavour flavour = sender_options.make_flavour(opts.protocol);
        spead2::send::stream_config config = sender_options.make_stream_config();

        /* Build the heaps */
        std::vector<spead2::send::heap> heaps;
        for (std::int64_t i = 0; i < num_heaps; i++)
        {
            heaps.emplace_back(flavour);
            heaps.back().add_item(0x1234, data, false);
        }
        heaps.emplace_back(flavour);
        heaps.back().add_end();

        /* Send the heaps */
        auto start = std::chrono::high_resolution_clock::now();
        std::int64_t transferred;
        std::unique_ptr<spead2::send::stream> stream = sender_options.make_stream(
            thread_pool.get_io_service(),
            opts.protocol,
            {endpoint},
            {{data.data(), data.size() * sizeof(data[0])}});
        sender s(*stream, heaps, opts);
        transferred = s.run();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_duration = end - start;
        double actual_rate = transferred / elapsed_duration.count();

        /* Get results */
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        control << "stop" << std::endl;
        std::int64_t received_heaps;
        control >> received_heaps;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        control.close();
        bool good = (received_heaps >= required_heaps);
        return std::make_pair(good, actual_rate);
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

static std::pair<bool, double> measure_connection(
    const options &opts, double rate,
    std::int64_t num_heaps, std::int64_t required_heaps)
{
    bool good = true;
    double rate_sum = 0.0;
    const int passes = 5;
    for (int i = 0; i < passes; i++)
    {
        std::pair<bool, double> result = measure_connection_once(opts, rate, num_heaps, required_heaps);
        if (!result.first)
            good = false;
        rate_sum += result.second;
    }
    return std::make_pair(good, rate_sum / passes);
}

static void main_master(int argc, const char **argv)
{
    options opts = parse_args(argc, argv, command_mode::MASTER);
    double best_actual = 0.0;

    /* Send 1GB as fast as possible to find an upper bound - receive rate
     * does not matter. Also do a warmup run first to warm up the receiver.
     */
    std::int64_t num_heaps = std::int64_t(1e9 / opts.heap_size) + 2;
    measure_connection_once(opts, 0.0, num_heaps, 0); // warmup
    std::pair<bool, double> result = measure_connection(opts, 0.0, num_heaps, num_heaps - 1);
    if (result.first)
    {
        if (!opts.quiet)
            std::cout << "Limited by send spead\n";
        best_actual = result.second;
    }
    else
    {
        if (!opts.quiet)
            std::cout << boost::format("Send rate: %.3f Gbps\n") % (result.second * 8e-9);

        double low = 0.0;
        double high = result.second;
        while (high - low > high * 0.02)
        {
            // Need at least 1GB of data to overwhelm cache effects, and want at least
            // 1 second for warmup effects.
            double rate = (low + high) * 0.5;
            num_heaps = std::int64_t(std::max(1e9, rate) / opts.heap_size) + 2;
            result = measure_connection(opts, rate, num_heaps, num_heaps - 1);
            bool good = result.first;
            double actual_rate = result.second;
            if (!opts.quiet)
                std::cout << boost::format("Rate: %.3f Gbps (%.3f actual): %s\n")
                    % (rate * 8e-9) % (actual_rate * 8e-9) % (good ? "GOOD" : "BAD");
            if (good)
            {
                low = rate;
                best_actual = actual_rate;
            }
            else
                high = rate;
        }
    }
    double rate_gbps = best_actual * 8e-9;
    if (opts.quiet)
        std::cout << rate_gbps << '\n';
    else
        std::cout << boost::format("Sustainable rate: %.3f Gbps\n") % rate_gbps;
}

class recv_stream : public spead2::recv::stream
{
private:
    virtual void heap_ready(spead2::recv::live_heap &&live) override
    {
        if (live.is_contiguous())
        {
            spead2::recv::heap heap(std::move(live));
            num_heaps++;
        }
    }

    virtual void stop_received() override
    {
        spead2::recv::stream::stop_received();
        stopped_promise.set_value();
    }

public:
    using spead2::recv::stream::stream;
    std::int64_t num_heaps = 0;
    std::promise<void> stopped_promise;
};

class recv_connection
{
protected:
    spead2::thread_pool thread_pool;

public:
    virtual std::int64_t stop(bool force) = 0;
    virtual spead2::recv::stream &get_stream() = 0;
    virtual ~recv_connection() {}
};

class recv_connection_callback : public recv_connection
{
private:
    recv_stream stream;

public:
    explicit recv_connection_callback(const options &opts)
        : stream(thread_pool, opts.receiver.make_stream_config(opts.protocol))
    {
    }

    virtual spead2::recv::stream &get_stream() override { return stream; }

    virtual std::int64_t stop(bool force) override
    {
        if (force)
            stream.stop();
        stream.stopped_promise.get_future().get();
        return stream.num_heaps;
    }
};

class recv_connection_ring : public recv_connection
{
private:
    spead2::recv::ring_stream<> stream;
    std::thread consumer;
    std::int64_t num_heaps = 0;

public:
    explicit recv_connection_ring(const options &opts)
        : stream(thread_pool,
                 opts.receiver.make_stream_config(opts.protocol),
                 opts.receiver.make_ring_stream_config())
    {
        consumer = std::thread([this] ()
        {
            try
            {
                while (true)
                {
                    stream.pop();
                    num_heaps++;
                }
            } catch (spead2::ringbuffer_stopped &e)
            {
            }
        });
    }

    virtual spead2::recv::stream &get_stream() override { return stream; }

    virtual std::int64_t stop(bool force) override
    {
        if (force)
            stream.stop();
        consumer.join();
        return num_heaps;
    }
};

static void main_agent(int argc, const char **argv)
{
    using asio::ip::tcp;
    using asio::ip::udp;
    options agent_opts = parse_args(argc, argv, command_mode::AGENT);
    spead2::thread_pool thread_pool;

    /* Look up the bind address for the control socket */
    tcp::resolver tcp_resolver(thread_pool.get_io_service());
    tcp::resolver::query tcp_query("0.0.0.0", agent_opts.endpoint);
    tcp::endpoint tcp_endpoint = *tcp_resolver.resolve(tcp_query);
    tcp::acceptor acceptor(thread_pool.get_io_service(), tcp_endpoint);
    while (true)
    {
        tcp::iostream control;
        acceptor.accept(*control.rdbuf());
        std::unique_ptr<recv_connection> connection;
        while (true)
        {
            std::string line;
            std::getline(control, line);
            if (!control)
                break;
            if (line == "start")
            {
                std::stringstream config_file;
                while (control)
                {
                    std::getline(control, line);
                    if (line == "end_config")
                        break;
                    config_file << line << '\n';
                }
                if (connection)
                {
                    std::cerr <<  "Start received while already running\n";
                    continue;
                }

                options opts;
                po::options_description desc;
                po::variables_map vm;
                opts.enumerate_wire(spead2::option_adder(desc));
                config_file.seekg(0);
                po::store(parse_config_file(config_file, desc), vm);
                vm.notify();   // Writes the options into opts
                std::string endpoint =
                    !opts.multicast.empty() ? opts.multicast : agent_opts.endpoint;
                opts.protocol.notify();
                opts.receiver.notify(opts.protocol);
                if (opts.receiver.ring)
                    connection.reset(new recv_connection_ring(opts));
                else
                    connection.reset(new recv_connection_callback(opts));
                opts.receiver.add_readers(connection->get_stream(), {endpoint}, opts.protocol, false);
                control << "ready" << std::endl;
            }
            else if (line == "stop")
            {
                if (!connection)
                {
                    std::cerr << "Stop received when already stopped\n";
                    continue;
                }
                auto num_heaps = connection->stop(true);
                connection.reset();
                std::cerr << num_heaps << '\n';
                control << num_heaps << std::endl;
            }
            else if (line == "exit")
                break;
            else
                std::cerr << "Bad command: " << line << '\n';
        }
    }
}

static void build_streambuf(std::streambuf &streambuf, const options &opts, std::int64_t num_heaps)
{
    spead2::thread_pool thread_pool;
    spead2::flavour flavour = opts.sender.make_flavour(opts.protocol);
    spead2::send::streambuf_stream stream(
        thread_pool.get_io_service(), streambuf,
        opts.sender.make_stream_config());
    spead2::send::heap heap(flavour), end_heap(flavour);
    std::vector<std::uint8_t> data(opts.heap_size);
    heap.add_item(0x1234, data, false);
    end_heap.add_end();
    for (std::int64_t i = 0; i < num_heaps; i++)
    {
        boost::system::error_code last_error;
        auto callback = [&last_error] (const boost::system::error_code &ec, spead2::item_pointer_t bytes)
        {
            if (!ec)
                last_error = ec;
        };
        stream.async_send_heap(i < num_heaps - 1 ? heap : end_heap, callback);
        stream.flush();
        if (last_error)
            throw boost::system::system_error(last_error);
    }
}

static void main_mem(int argc, const char **argv)
{
    options opts = parse_args(argc, argv, command_mode::MEM);
    // Use about 1GiB of data
    std::int64_t num_heaps = 1024 * 1024 * 1024 / opts.heap_size + 1;
    std::string data;
    {
        std::stringstream ss;
        build_streambuf(*ss.rdbuf(), opts, num_heaps);
        data = ss.str();
    }

    spead2::thread_pool thread_pool;
    std::unique_ptr<recv_connection> connection;
    std::chrono::high_resolution_clock::time_point start, end;
    double elapsed = 0.0;
    const int passes = 100;
    for (int pass = 0; pass < passes; pass++)
    {
        if (opts.receiver.ring)
            connection.reset(new recv_connection_ring(opts));
        else
            connection.reset(new recv_connection_callback(opts));
        start = std::chrono::high_resolution_clock::now();
        connection->get_stream().emplace_reader<spead2::recv::mem_reader>(
            (const std::uint8_t *) data.data(), data.size());
        connection->stop(false);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_duration = end - start;
        elapsed += elapsed_duration.count();
    }
    double rate = data.size() * passes / elapsed;
    double rate_gbps = rate * 8e-9;
    if (!opts.quiet)
    {
        std::cout << "Transferred " << data.size() << " bytes in " << elapsed << " seconds\n";
        std::cout << rate_gbps << " Gbps\n";
    }
    else
        std::cout << rate_gbps << '\n';
}

int main(int argc, const char **argv)
{
    if (argc >= 2 && argv[1] == std::string("master"))
        main_master(argc - 1, argv + 1);
    else if (argc >= 2 && argv[1] == std::string("agent"))
        main_agent(argc - 1, argv + 1);
    else if (argc >= 2 && argv[1] == std::string("mem"))
        main_mem(argc - 1, argv + 1);
    else
    {
        std::cerr << "Usage:\n"
            << "    spead2_bench master <host>:<port> [options]\n"
            << "OR\n"
            << "    spead2_bench agent <port> [options]\n"
            << "OR\n"
            << "    spead2_bench mem [options]\n";
        return 2;
    }

    return 0;
}
