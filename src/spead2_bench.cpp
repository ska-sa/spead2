/* Copyright 2015, 2017, 2019 SKA South Africa
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

namespace po = boost::program_options;
namespace asio = boost::asio;

struct options
{
    bool quiet = false;
    bool ring = false;
    bool memcpy_nt = false;
    std::size_t packet_size = 9172;
    std::size_t heap_size = 4194304;
    int heap_address_bits = 40;
    std::size_t send_buffer = spead2::send::udp_stream::default_buffer_size;
    std::size_t recv_buffer = spead2::recv::udp_reader::default_buffer_size;
    std::size_t burst_size = spead2::send::stream_config::default_burst_size;
    double burst_rate_ratio = spead2::send::stream_config::default_burst_rate_ratio;
    std::size_t heaps = spead2::recv::stream_base::default_max_heaps;
    std::size_t ring_heaps = spead2::recv::ring_stream_base::default_ring_heaps;
    std::size_t mem_max_free = 12;
    std::size_t mem_initial = 8;

    std::string send_ibv_if;
    int send_ibv_comp_vector = 0;
    int send_ibv_max_poll =
#if SPEAD2_USE_IBV
        spead2::send::udp_ibv_stream::default_max_poll;
#else
        0;
#endif
    std::string recv_ibv_if;
    int recv_ibv_comp_vector = 0;
    int recv_ibv_max_poll =
#if SPEAD2_USE_IBV
        spead2::recv::udp_ibv_reader::default_max_poll;
#else
        0;
#endif

    std::string multicast;
    std::string host;
    std::string port;
};

enum class command_mode
{
    MASTER,
    SLAVE,
    MEM
};

template<typename T>
static po::typed_value<T> *make_opt(T &var)
{
    return po::value<T>(&var)->default_value(var);
}

static void usage(std::ostream &o, const po::options_description &desc, command_mode mode)
{
    switch (mode)
    {
    case command_mode::MASTER:
        o << "Usage: spead2_bench master <host> <port> [options]\n";
        break;
    case command_mode::SLAVE:
        o << "Usage: spead2_bench slave <port>\n";
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
    if (mode == command_mode::MASTER || mode == command_mode::MEM)
    {
        desc.add_options()
            ("quiet", po::bool_switch(&opts.quiet)->default_value(opts.quiet), "Print only the final result")
            ("ring", po::bool_switch(&opts.ring)->default_value(opts.ring), "Use a ring buffer for heaps")
            ("memcpy-nt", po::bool_switch(&opts.memcpy_nt)->default_value(opts.memcpy_nt), "Use non-temporal memcpy")
            ("packet", make_opt(opts.packet_size), "Maximum packet size to use for UDP")
            ("heap-size", make_opt(opts.heap_size), "Payload size for heap")
            ("addr-bits", make_opt(opts.heap_address_bits), "Heap address bits")
            ("heaps", make_opt(opts.heaps), "Maximum number of in-flight heaps")
            ("ring-heaps", make_opt(opts.ring_heaps), "Ring buffer capacity in heaps")
            ("mem-max-free", make_opt(opts.mem_max_free), "Maximum free memory buffers")
            ("mem-initial", make_opt(opts.mem_initial), "Initial free memory buffers");
    }
    if (mode == command_mode::MASTER)
    {
        desc.add_options()
            ("send-buffer", make_opt(opts.send_buffer), "Socket buffer size (sender)")
            ("recv-buffer", make_opt(opts.recv_buffer), "Socket buffer size (receiver)")
            ("burst", make_opt(opts.burst_size), "Send burst size")
            ("burst-rate-ratio", make_opt(opts.burst_rate_ratio), "Hard rate limit, relative to the nominal rate")
            ("multicast", make_opt(opts.multicast), "Multicast group to use, instead of unicast")
#if SPEAD2_USE_IBV
            ("send-ibv", make_opt(opts.send_ibv_if), "Interface address for ibverbs (sender)")
            ("send-ibv-vector", make_opt(opts.send_ibv_comp_vector), "Interrupt vector (-1 for polled) (sender)")
            ("send-ibv-max-poll", make_opt(opts.send_ibv_max_poll), "Maximum number of times to poll in a row (sender)")
            ("recv-ibv", make_opt(opts.recv_ibv_if), "Interface address for ibverbs (receiver)")
            ("recv-ibv-vector", make_opt(opts.recv_ibv_comp_vector), "Interrupt vector (-1 for polled) (receiver)")
            ("recv-ibv-max-poll", make_opt(opts.recv_ibv_max_poll), "Maximum number of times to poll in a row (receiver)")
#endif
        ;
        hidden.add_options()
            ("host", po::value<std::string>(&opts.host));
    }
    desc.add_options()
        ("help,h", "Show help text");
    if (mode == command_mode::MASTER || mode == command_mode::SLAVE)
    {
        hidden.add_options()
            ("port", po::value<std::string>(&opts.port));
    }
    all.add(desc);
    all.add(hidden);

    po::positional_options_description positional;
    switch (mode)
    {
    case command_mode::MASTER:
        positional.add("host", 1);
        positional.add("port", 1);
        break;
    case command_mode::SLAVE:
        positional.add("port", 1);
        break;
    case command_mode::MEM:
        break;
    }
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
        if ((mode == command_mode::MASTER || mode == command_mode::SLAVE)
            && !vm.count("port"))
        {
            throw po::error("too few positional options have been specified on the command line");
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

// Simple encoding scheme to allow empty strings to be passed in text
static std::string encode_string(const std::string &s)
{
    std::ostringstream encoder;
    encoder.imbue(std::locale::classic());
    encoder << '+';
    encoder << std::hex << std::setw(2) << std::setfill('0');
    for (unsigned char c : s)
        encoder << int(c);
    return encoder.str();
}

static std::string decode_string(const std::string &s)
{
    if (s.empty() || s[0] != '+')
        throw std::invalid_argument("string was not encoded");
    std::string out;
    for (std::size_t i = 1; i + 1 < s.size(); i += 2)
    {
        std::istringstream decoder(s.substr(i, 2));
        decoder.imbue(std::locale::classic());
        int value;
        decoder >> std::hex >> value;
        out += (unsigned char) value;
    }
    return out;
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
    : stream(stream), heaps(heaps), max_heaps(std::min(opts.heaps, heaps.size()))
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
    asio::io_service io_service;
    udp::resolver resolver(io_service);
    udp::resolver::query query(opts.multicast.empty() ? opts.host : opts.multicast, opts.port);
    udp::endpoint endpoint = *resolver.resolve(query);

    /* Initiate the control connection */
    tcp::iostream control(opts.host, opts.port);
    try
    {
        control.exceptions(std::ios::failbit);
        control << "start "
            << int(opts.ring) << ' '
            << int(opts.memcpy_nt) << ' '
            << opts.packet_size << ' '
            << opts.heap_size << ' '
            << opts.heap_address_bits << ' '
            << opts.recv_buffer << ' '
            << opts.burst_size << ' '
            << opts.burst_rate_ratio << ' '
            << opts.heaps << ' '
            << opts.ring_heaps << ' '
            << opts.mem_max_free << ' '
            << opts.mem_initial << ' '
            << encode_string(opts.multicast) << ' '
            << encode_string(opts.recv_ibv_if) << ' '
            << opts.recv_ibv_comp_vector << ' '
            << opts.recv_ibv_max_poll << std::endl;
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
        spead2::send::stream_config config(
            opts.packet_size, rate, opts.burst_size, opts.heaps, opts.burst_rate_ratio);

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
        std::unique_ptr<spead2::send::stream> stream;
#if SPEAD2_USE_IBV
        if (opts.send_ibv_if != "")
        {
            boost::asio::ip::address interface_address =
                boost::asio::ip::address::from_string(opts.send_ibv_if);
            stream.reset(new spead2::send::udp_ibv_stream(
                thread_pool.get_io_service(), endpoint, config, interface_address,
                opts.send_buffer, 1, opts.send_ibv_comp_vector, opts.send_ibv_max_poll));
        }
        else
#endif
        {
            stream.reset(new spead2::send::udp_stream(
                thread_pool.get_io_service(), endpoint, config, opts.send_buffer));
        }
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
    std::shared_ptr<spead2::memory_pool> memory_pool;

    explicit recv_connection(const options &opts)
        : thread_pool(),
        memory_pool(std::make_shared<spead2::memory_pool>(opts.heap_size, opts.heap_size + 1024, opts.mem_max_free, opts.mem_initial))
    {
    }

public:
    virtual std::int64_t stop(bool force) = 0;
    virtual void emplace_udp_reader(const boost::asio::ip::udp::endpoint &endpoint,
                                    const options &opts) = 0;
    virtual void emplace_mem_reader(const std::uint8_t *ptr, std::size_t length) = 0;
    virtual ~recv_connection() {}
};

template<typename Stream>
static void emplace_udp_reader(Stream &stream, const boost::asio::ip::udp::endpoint &endpoint,
                               const options &opts)
{
    if (opts.recv_ibv_if != "")
    {
#if SPEAD2_USE_IBV
        boost::asio::ip::address interface_address =
            boost::asio::ip::address::from_string(opts.recv_ibv_if);
        stream.template emplace_reader<spead2::recv::udp_ibv_reader>(
            endpoint, interface_address, opts.packet_size, opts.recv_buffer,
            opts.recv_ibv_comp_vector, opts.recv_ibv_max_poll);
#else
        std::cerr << "--recv-ibv passed but slave does not support ibv\n";
        std::exit(1);
#endif
    }
    else
    {
        stream.template emplace_reader<spead2::recv::udp_reader>(
            endpoint, opts.packet_size, opts.recv_buffer);
    }
}

class recv_connection_callback : public recv_connection
{
    recv_stream stream;

public:
    explicit recv_connection_callback(const options &opts)
        : recv_connection(opts), stream(thread_pool, 0, opts.heaps)
    {
        stream.set_memory_allocator(memory_pool);
        if (opts.memcpy_nt)
            stream.set_memcpy(spead2::MEMCPY_NONTEMPORAL);
    }

    template<typename Reader, typename... Args>
    void emplace_reader(Args&&... args)
    {
        stream.emplace_reader<Reader>(std::forward<Args>(args)...);
    }

    virtual void emplace_udp_reader(const boost::asio::ip::udp::endpoint &endpoint,
                                    const options &opts) override
    {
        ::emplace_udp_reader(stream, endpoint, opts);
    }

    virtual void emplace_mem_reader(const std::uint8_t *ptr, std::size_t length) override
    {
        stream.emplace_reader<spead2::recv::mem_reader>(ptr, length);
    }

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
    spead2::recv::ring_stream<> stream;
    std::thread consumer;
    std::int64_t num_heaps = 0;

public:
    explicit recv_connection_ring(const options &opts)
        : recv_connection(opts), stream(thread_pool, 0, opts.heaps, opts.ring_heaps)
    {
        stream.set_memory_allocator(memory_pool);
        if (opts.memcpy_nt)
            stream.set_memcpy(spead2::MEMCPY_NONTEMPORAL);
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

    virtual void emplace_udp_reader(const boost::asio::ip::udp::endpoint &endpoint,
                                    const options &opts) override
    {
        ::emplace_udp_reader(stream, endpoint, opts);
    }

    virtual void emplace_mem_reader(const std::uint8_t *ptr, std::size_t length) override
    {
        stream.emplace_reader<spead2::recv::mem_reader>(ptr, length);
    }

    virtual std::int64_t stop(bool force) override
    {
        if (force)
            stream.stop();
        consumer.join();
        return num_heaps;
    }
};

static void main_slave(int argc, const char **argv)
{
    using asio::ip::tcp;
    using asio::ip::udp;
    options slave_opts = parse_args(argc, argv, command_mode::SLAVE);
    spead2::thread_pool thread_pool;

    /* Look up the bind address for the control socket */
    tcp::resolver tcp_resolver(thread_pool.get_io_service());
    tcp::resolver::query tcp_query("0.0.0.0", slave_opts.port);
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
                toks >> opts.ring
                    >> opts.memcpy_nt
                    >> opts.packet_size
                    >> opts.heap_size
                    >> opts.heap_address_bits
                    >> opts.recv_buffer
                    >> opts.burst_size
                    >> opts.burst_rate_ratio
                    >> opts.heaps
                    >> opts.ring_heaps
                    >> opts.mem_max_free
                    >> opts.mem_initial
                    >> opts.multicast
                    >> opts.recv_ibv_if
                    >> opts.recv_ibv_comp_vector
                    >> opts.recv_ibv_max_poll;
                opts.multicast = decode_string(opts.multicast);
                opts.recv_ibv_if = decode_string(opts.recv_ibv_if);
                /* Look up the bind address for the data socket */
                udp::resolver resolver(thread_pool.get_io_service());
                udp::resolver::query query(opts.multicast.empty() ? "0.0.0.0" : opts.multicast,
                                           slave_opts.port);
                udp::endpoint endpoint = *resolver.resolve(query);
                if (opts.ring)
                    connection.reset(new recv_connection_ring(opts));
                else
                    connection.reset(new recv_connection_callback(opts));
                connection->emplace_udp_reader(endpoint, opts);
                control << "ready" << std::endl;
            }
            else if (cmd == "stop")
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
            else if (cmd == "exit")
                break;
            else
                std::cerr << "Bad command: " << line << '\n';
        }
    }
}

static void build_streambuf(std::streambuf &streambuf, const options &opts, std::int64_t num_heaps)
{
    spead2::thread_pool thread_pool;
    spead2::flavour flavour(4, 64, opts.heap_address_bits);
    spead2::send::stream_config config(opts.packet_size);
    spead2::send::streambuf_stream stream(thread_pool.get_io_service(), streambuf, config);
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
        if (opts.ring)
            connection.reset(new recv_connection_ring(opts));
        else
            connection.reset(new recv_connection_callback(opts));
        start = std::chrono::high_resolution_clock::now();
        connection->emplace_mem_reader((const std::uint8_t *) data.data(), data.size());
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
    else if (argc >= 2 && argv[1] == std::string("slave"))
        main_slave(argc - 1, argv + 1);
    else if (argc >= 2 && argv[1] == std::string("mem"))
        main_mem(argc - 1, argv + 1);
    else
    {
        std::cerr << "Usage:\n"
            << "    spead2_bench master <host> <port> [options]\n"
            << "OR\n"
            << "    spead2_bench slave <port> [options]\n"
            << "OR\n"
            << "    spead2_bench mem [options]\n";
        return 2;
    }

    return 0;
}
