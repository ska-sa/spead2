/* Copyright 2016-2019 SKA South Africa
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
 * Utility program to dump raw multicast packets, using ibverbs. It works with
 * any multicast UDP data, not just SPEAD.
 *
 * The design is based on three threads:
 * 1. A network thread that handles the ibverbs interface, removing packets
 *    from the completion queue and passing them (in batches) to the collector
 *    thread. It is also responsible for periodically reporting rates.
 * 2. A collector thread that receives batches of packets and copies them
 *    into new page-aligned buffers.
 * 3. A disk thread that writes page-aligned buffers to disk (if requested,
 *    using O_DIRECT). There can actually be more than one of these.
 *
 * If no filename is given, the latter two threads are disabled.
 */

#include <spead2/common_ibv.h>
#include <spead2/common_raw_packet.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_logging.h>
#include <spead2/common_memory_pool.h>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <functional>
#include <atomic>
#include <cstring>
#include <future>
#include <thread>
#include <unistd.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>

namespace po = boost::program_options;

typedef std::chrono::high_resolution_clock::time_point time_point;

static const char * const level_names[] =
{
    "warning",
    "info",
    "debug"
};

static void log_function(spead2::log_level level, const std::string &msg)
{
    unsigned int level_idx = static_cast<unsigned int>(level);
    assert(level_idx < sizeof(level_names) / sizeof(level_names[0]));
    std::cerr << level_names[level_idx] << ": " << msg << "\n";
}

struct options
{
    std::vector<std::string> endpoints;
    std::string interface;
    std::string filename;
    int snaplen = 9230;
    std::size_t net_buffer = 128 * 1024 * 1024;
    std::size_t disk_buffer = 64 * 1024 * 1024;
    int network_affinity = -1;
    int collect_affinity = -1;
    std::vector<int> disk_affinity;
    bool quiet = false;
    std::uint64_t count = std::numeric_limits<std::uint64_t>::max();
#ifdef O_DIRECT
    bool direct = false;
#endif
};

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: mcdump [options] -i <iface-addr> <filename> <group>:<port>...\n";
    o << "Use - as filename to skip the write and just count.\n";
    o << desc;
}

template<typename T>
static po::typed_value<T> *make_opt(T &var)
{
    return po::value<T>(&var)->default_value(var);
}

template<typename T>
static po::typed_value<T> *make_opt_nodefault(T &var)
{
    return po::value<T>(&var);
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
        ("interface,i", make_opt_nodefault(opts.interface), "IP address of capture interface")
        ("snaplen,s", make_opt(opts.snaplen), "Maximum frame size to capture")
        ("net-buffer", make_opt(opts.net_buffer), "Maximum memory for buffering packets from the network")
        ("disk-buffer", make_opt(opts.disk_buffer), "Maximum memory for buffering bytes to disk")
        ("network-cpu,N", make_opt(opts.network_affinity), "CPU core for network receive")
        ("collect-cpu,C", make_opt(opts.collect_affinity), "CPU core for rearranging data")
        ("disk-cpu,D", po::value<std::vector<int>>(&opts.disk_affinity)->composing(), "CPU core for disk writing (can be used multiple times)")
        ("quiet,q", make_opt(opts.quiet), "Do not report counts and rates while running")
        ("count,c", make_opt(opts.count), "Stop after this many packets")
#ifdef O_DIRECT
        ("direct-io", make_opt(opts.direct), "Use O_DIRECT I/O (not supported on all filesystems)")
#endif
        ("help,h", "Show help text")
    ;

    hidden.add_options()
        ("filename", make_opt_nodefault(opts.filename), "output filename")
        ("endpoint", po::value<std::vector<std::string>>(&opts.endpoints)->composing(), "multicast-group:port")
    ;
    all.add(desc);
    all.add(hidden);

    po::positional_options_description positional;
    positional.add("filename", 1);
    positional.add("endpoint", -1);
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
        if (!vm.count("filename") || !vm.count("endpoint"))
            throw po::error("too few positional options have been specified on the command line");
        if (!vm.count("interface"))
            throw po::error("interface IP address (-i) is required");
        return opts;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        usage(std::cerr, desc);
        std::exit(2);
    }
}

// pcap file header: see https://wiki.wireshark.org/Development/LibpcapFileFormat
struct file_header
{
    std::uint32_t magic_number = 0xa1b23c4d;
    std::uint16_t version_major = 2;
    std::uint16_t version_minor = 4;
    std::int32_t this_zone = 0;
    std::uint32_t sigfigs = 0;
    std::uint32_t snaplen;
    std::uint32_t network = 1; // DLT_EN10MB
};

// pcap record header: see https://wiki.wireshark.org/Development/LibpcapFileFormat
struct record_header
{
    std::uint32_t ts_sec = 0;
    std::uint32_t ts_nsec = 0;
    std::uint32_t incl_len;
    std::uint32_t orig_len;
};

/* Data associated with a single packet. When using a multi-packet receive
 * queue, wr and sg are not used.
 */
struct chunk_entry
{
    ibv_recv_wr wr;
    ibv_sge sg;
    record_header record;
};

/* A contiguous chunk of memory holding packets, before collection. */
struct chunk
{
    std::uint32_t n_records;
    std::size_t n_bytes;
    bool full;                  ///< No more work requests referencing it
    std::unique_ptr<chunk_entry[]> entries;
    std::unique_ptr<iovec[]> iov;
    spead2::memory_pool::pointer storage;
    spead2::ibv_mr_t storage_mr;
};

static std::atomic<bool> stop{false};

static void signal_handler(int)
{
    stop = true;
}

class writer
{
private:
    struct buffer
    {
        off_t offset; // offset in file
        spead2::memory_allocator::pointer data;
        std::size_t length;
    };

    static constexpr std::size_t buffer_size = 8 * 1024 * 1024;

    int fd;
    std::size_t depth;
    spead2::ringbuffer<buffer> ring, free_ring;
    buffer cur_buffer;
    off_t total_bytes = 0;
    std::vector<std::future<void>> writer_threads;

    static std::size_t compute_depth(const options &opts);
    void flush();
    void writer_thread(int affinity);

public:
    writer(const options &opts, int fd, spead2::memory_allocator &allocator);
    ~writer();
    void close();

    void write(const void *data, std::size_t length);
};

constexpr std::size_t writer::buffer_size;

std::size_t writer::compute_depth(const options &opts)
{
    std::size_t depth = opts.disk_buffer / buffer_size;
    if (depth == 0)
        depth = 1;
    return depth;
}

writer::writer(const options &opts, int fd, spead2::memory_allocator &allocator)
    : fd(fd), depth(compute_depth(opts)), ring(depth), free_ring(depth)
{
    for (std::size_t i = 0; i < depth; i++)
    {
        buffer b;
        b.data = allocator.allocate(buffer_size, nullptr);
        b.length = 0;
        b.offset = 0;
        free_ring.push(std::move(b));
    }
    cur_buffer = free_ring.pop();
    if (opts.disk_affinity.empty())
        writer_threads.push_back(std::async(std::launch::async, [this] { writer_thread(-1); }));
    else
    {
        for (int affinity : opts.disk_affinity)
        {
            writer_threads.push_back(std::async(std::launch::async,
                [this, affinity] { writer_thread(affinity); }));
        }
    }
}

writer::~writer()
{
    /* We don't try to close the file. While this violates RAII, it avoids
     * double-exception problems. We only get here with the file still open in
     * error cases, in which case it is quite likely that flushing the file
     * will fail anyway (e.g., because the disk is full); and the process is
     * about to die anyway.
     */
}

void writer::close()
{
    if (fd == -1)
        return;
    if (cur_buffer.length > 0)
        flush();
    // Shut down the writer threads
    ring.stop();
    for (std::future<void> &future : writer_threads)
        future.get();
    writer_threads.clear();
    // free memory
    free_ring.stop();
    while (true)
    {
        try
        {
            free_ring.pop();
        }
        catch (spead2::ringbuffer_stopped &)
        {
            break;
        }
    }
    // flush always writes the entire buffer (possibly necessary for O_DIRECT),
    // so truncate back to the actual desired size
    if (ftruncate(fd, total_bytes) != 0)
        spead2::throw_errno("ftruncate failed");
    if (::close(fd) != 0)
        spead2::throw_errno("close failed");
    fd = -1;
}

void writer::write(const void *data, std::size_t length)
{
    while (length > 0)
    {
        std::size_t n = std::min(length, buffer_size - cur_buffer.length);
        std::memcpy(cur_buffer.data.get() + cur_buffer.length, data, n);
        data = (const std::uint8_t *) data + n;
        length -= n;
        cur_buffer.length += n;
        if (cur_buffer.length == buffer_size)
            flush();
    }
}

void writer::flush()
{
    total_bytes += cur_buffer.length;
    ring.push(std::move(cur_buffer));
    cur_buffer = free_ring.pop();
    cur_buffer.offset = total_bytes;
}

void writer::writer_thread(int affinity)
{
    try
    {
        if (affinity >= 0)
            spead2::thread_pool::set_affinity(affinity);
        while (true)
        {
            try
            {
                buffer b = ring.pop();
                std::uint8_t *ptr = b.data.get();
                std::size_t remain = buffer_size;
                off_t offset = b.offset;
                while (remain > 0)
                {
                    ssize_t ret = pwrite(fd, ptr, remain, offset);
                    if (ret < 0)
                    {
                        if (errno == EINTR)
                            continue;
                        spead2::throw_errno("write failed");
                    }
                    ptr += ret;
                    remain -= ret;
                    offset += ret;
                }
                b.length = 0;
                free_ring.push(std::move(b));
            }
            catch (spead2::ringbuffer_stopped &)
            {
                break;
            }
        }
    }
    catch (std::exception &e)
    {
        stop = true;
        throw;
    }
}

struct chunking_scheme
{
    std::size_t max_records;   ///< Packets per chunk
    std::size_t n_chunks;      ///< Number of chunks
    std::size_t chunk_size;    ///< Bytes per chunk
};

typedef std::function<chunking_scheme(const options &,
                                      const spead2::rdma_cm_id_t &)> chunking_scheme_generator;

/* Joints multicast groups for its lifetime */
class joiner
{
private:
    boost::asio::io_service io_service;
    boost::asio::ip::udp::socket join_socket;

public:
    joiner(const boost::asio::ip::address_v4 &interface_address,
           const std::vector<boost::asio::ip::udp::endpoint> &endpoints);
};

joiner::joiner(const boost::asio::ip::address_v4 &interface_address,
               const std::vector<boost::asio::ip::udp::endpoint> &endpoints)
    : join_socket(io_service, endpoints[0].protocol())
{
    join_socket.set_option(boost::asio::socket_base::reuse_address(true));
    for (const auto &endpoint : endpoints)
        join_socket.set_option(boost::asio::ip::multicast::join_group(
            endpoint.address().to_v4(), interface_address));
}

class capture_base
{
protected:
    typedef spead2::ringbuffer<chunk> ringbuffer;

    std::unique_ptr<writer> w;
    const options opts;

    boost::asio::ip::address_v4 interface_address;
    spead2::rdma_event_channel_t event_channel;
    spead2::rdma_cm_id_t cm_id;
    spead2::ibv_pd_t pd;
    const chunking_scheme chunking;

#if SPEAD2_USE_IBV_EXP
    bool timestamp_support = false;
    // To convert HW timestamps to nanoseconds
    double hca_ns_per_clock = 0.0;
    std::uint64_t timestamp_subtract = 0; // initial HW timestamp in ticks
    std::uint64_t timestamp_add = 0;      // initial system timestamp in ns
#endif

    ringbuffer ring;
    ringbuffer free_ring;
    std::uint64_t errors = 0;
    std::uint64_t packets = 0;
    std::uint64_t bytes = 0;
    time_point start_time;
    time_point last_report;
    std::uint64_t last_errors = 0;
    std::uint64_t last_packets = 0;
    std::uint64_t last_bytes = 0;

    void init_timestamp_support();
    chunk make_chunk(spead2::memory_allocator &allocator);

    void add_to_free(chunk &&c);
    void chunk_done(chunk &&c);

    void collect_thread();

    virtual void network_thread() = 0;
    virtual void post_chunk(chunk &c) = 0;
    virtual void init_record(chunk &c, std::size_t idx) = 0;

    void chunk_ready(chunk &&c);
    void report_rates(time_point now);

    capture_base(const options &opts, const chunking_scheme_generator &gen_chunking);

public:
    virtual ~capture_base();
    void run();
};

chunk capture_base::make_chunk(spead2::memory_allocator &allocator)
{
    const std::size_t max_records = chunking.max_records;
    const std::size_t chunk_size = chunking.chunk_size;
    chunk c;
    c.n_records = 0;
    c.n_bytes = 0;
    c.full = false;
    c.entries.reset(new chunk_entry[max_records]);
    c.iov.reset(new iovec[2 * max_records]);
    c.storage = allocator.allocate(chunk_size, nullptr);
    c.storage_mr = spead2::ibv_mr_t(pd, c.storage.get(), chunk_size, IBV_ACCESS_LOCAL_WRITE);
    std::uintptr_t ptr = (std::uintptr_t) c.storage.get();
    for (std::uint32_t i = 0; i < max_records; i++)
    {
        c.entries[i].wr.wr_id = i;
        c.entries[i].wr.next = (i + 1 < max_records) ? &c.entries[i + 1].wr : nullptr;
        c.entries[i].wr.num_sge = 1;
        c.entries[i].wr.sg_list = &c.entries[i].sg;
        c.entries[i].sg.addr = ptr;
        c.entries[i].sg.length = opts.snaplen;
        c.entries[i].sg.lkey = c.storage_mr->lkey;
        c.iov[2 * i].iov_base = &c.entries[i].record;
        c.iov[2 * i].iov_len = sizeof(record_header);
        c.iov[2 * i + 1].iov_base = (void *) ptr;
        ptr += opts.snaplen;
    }
    return c;
}

void capture_base::add_to_free(chunk &&c)
{
    c.n_records = 0;
    c.n_bytes = 0;
    c.full = false;
    post_chunk(c);
    free_ring.push(std::move(c));
}

void capture_base::chunk_done(chunk &&c)
{
    /* Only post a new receive if the chunk was full. If it was
     * not full, then this was the last chunk, and we're about to
     * get a stop. Some of the work requests are already in the
     * queue, so posting them again is asking for trouble.
     *
     * If the chunk was not full, we can't just free it, because
     * the QP might still be receiving data and writing it to the
     * chunk. So we push it back onto the ring without posting a
     * new receive, just to keep it live.
     */
    if (c.full)
        add_to_free(std::move(c));
    else
        free_ring.push(std::move(c));
}

void capture_base::chunk_ready(chunk &&c)
{
    if (opts.filename != "-")
        ring.push(std::move(c));
    else
        chunk_done(std::move(c));
}

void capture_base::collect_thread()
{
    try
    {
        if (opts.collect_affinity >= 0)
            spead2::thread_pool::set_affinity(opts.collect_affinity);

        file_header header;
        header.snaplen = opts.snaplen;
        w->write(&header, sizeof(header));
        while (true)
        {
            try
            {
                chunk c = ring.pop();
                std::uint32_t n_iov = 2 * c.n_records;
                for (std::uint32_t i = 0; i < n_iov; i++)
                    w->write(c.iov[i].iov_base, c.iov[i].iov_len);

                chunk_done(std::move(c));
            }
            catch (spead2::ringbuffer_stopped &)
            {
                free_ring.stop();
                w->close();
                break;
            }
        }
    }
    catch (std::exception &e)
    {
        stop = true;
        throw;
    }
}

void capture_base::report_rates(time_point now)
{
    boost::format formatter("B: %|-14| P: %|-12| MB/s: %|-10.0f| kP/s: %|-10.0f|\n");
    std::chrono::duration<double> elapsed = now - last_report;
    double byte_rate = (bytes - last_bytes) / elapsed.count();
    double packet_rate = (packets - last_packets) / elapsed.count();
    formatter % bytes % packets % (byte_rate / 1e6) % (packet_rate / 1e3);
    std::cout << formatter;
    last_bytes = bytes;
    last_packets = packets;
    last_errors = errors;
    last_report = now;
}

static boost::asio::ip::address_v4 get_interface_address(const options &opts)
{
    boost::asio::ip::address_v4 interface_address;
    try
    {
        interface_address = boost::asio::ip::address_v4::from_string(opts.interface);
    }
    catch (std::exception &)
    {
        throw std::runtime_error("Invalid interface address " + opts.interface);
    }
    return interface_address;
}

static spead2::rdma_cm_id_t create_cm_id(const boost::asio::ip::address_v4 &interface_address,
                                         const spead2::rdma_event_channel_t &event_channel)
{
    spead2::rdma_cm_id_t cm_id(event_channel, nullptr, RDMA_PS_UDP);
    cm_id.bind_addr(interface_address);
    return cm_id;
}

capture_base::capture_base(const options &opts, const chunking_scheme_generator &gen_chunking)
    : opts(opts),
    interface_address(get_interface_address(opts)),
    cm_id(create_cm_id(interface_address, event_channel)),
    pd(cm_id),
    chunking(gen_chunking(opts, cm_id)),
    ring(chunking.n_chunks),
    free_ring(chunking.n_chunks)
{
    init_timestamp_support();
}

capture_base::~capture_base()
{
    // This is needed (in the error case) to unblock the writer threads so that
    // shutdown doesn't deadlock.
    if (w)
        w->close();
}

void capture_base::init_timestamp_support()
{
#if SPEAD2_USE_IBV_EXP
    timestamp_support = true;
    // Get tick rate of the HW timestamp counter
    {
        ibv_exp_device_attr device_attr = {};
        device_attr.comp_mask = IBV_EXP_DEVICE_ATTR_WITH_HCA_CORE_CLOCK;
        int ret = ibv_exp_query_device(cm_id->verbs, &device_attr);
        /* There is some confusion about the units of hca_core_clock. Mellanox
         * documentation indicates it is in MHz
         * (http://www.mellanox.com/related-docs/prod_software/Mellanox_OFED_Linux_User_Manual_v4.1.pdf)
         * but their implementation in OFED 4.1 returns kHz. We work in kHz but if
         * if the value is suspiciously small (< 10 MHz) we assume the units were MHz.
         */
        if (device_attr.hca_core_clock < 10000)
            device_attr.hca_core_clock *= 1000;
        if (ret == 0 && (device_attr.comp_mask & IBV_EXP_DEVICE_ATTR_WITH_HCA_CORE_CLOCK))
            hca_ns_per_clock = 1e6 / device_attr.hca_core_clock;
        else
        {
            spead2::log_warning("could not query hca_core_clock: timestamps disabled");
            timestamp_support = false;
        }
    }
    // Get current value of HW timestamp counter, to subtract
    if (timestamp_support)
    {
        ibv_exp_values values = {};
        values.comp_mask = IBV_EXP_VALUES_HW_CLOCK;
        int ret = ibv_exp_query_values(cm_id->verbs, IBV_EXP_VALUES_HW_CLOCK, &values);
        if (ret == 0 && (values.comp_mask & IBV_EXP_VALUES_HW_CLOCK))
        {
            timestamp_subtract = values.hwclock;
            timespec now;
            clock_gettime(CLOCK_REALTIME, &now);
            timestamp_add = now.tv_sec * std::uint64_t(1000000000) + now.tv_nsec;
        }
        else
        {
            spead2::log_warning("could not query current HW time: timestamps disabled");
            timestamp_support = false;
        }
    }
#endif
}

static boost::asio::ip::udp::endpoint make_endpoint(const std::string &s)
{
    // Use rfind rather than find because IPv6 addresses contain :'s
    auto pos = s.rfind(':');
    if (pos == std::string::npos)
    {
        throw std::runtime_error("Endpoint " + s + " is missing a port number");
    }
    try
    {
        boost::asio::ip::address_v4 addr = boost::asio::ip::address_v4::from_string(s.substr(0, pos));
        if (!addr.is_multicast())
            throw std::runtime_error("Address " + s.substr(0, pos) + " is not a multicast address");
        std::uint16_t port = boost::lexical_cast<std::uint16_t>(s.substr(pos + 1));
        return boost::asio::ip::udp::endpoint(addr, port);
    }
    catch (boost::bad_lexical_cast &)
    {
        throw std::runtime_error("Invalid port number " + s.substr(pos + 1));
    }
}

void capture_base::run()
{
    using boost::asio::ip::udp;

    std::shared_ptr<spead2::mmap_allocator> allocator =
        std::make_shared<spead2::mmap_allocator>(0, true);
    bool has_file = opts.filename != "-";
    if (has_file)
    {
        int fd_flags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_DIRECT
        if (opts.direct)
            fd_flags |= O_DIRECT;
#endif
        int fd = open(opts.filename.c_str(), fd_flags, 0666);
        if (fd < 0)
            spead2::throw_errno("open failed");
        w.reset(new writer(opts, fd, *allocator));
    }

    /* Run in a thread that's bound to network_affinity, so that the pages are more likely
     * to end up on the right NUMA node. It is system policy dependent, the
     * Linux default is to allocate memory on the same node as the CPU that
     * made the allocation.
     *
     * We don't want to bind the parent thread, because we haven't yet forked
     * off collect_thread, and if it doesn't have a specific affinity we don't
     * want it to inherit network_affinity.
     */
    std::future<void> alloc_future = std::async(std::launch::async, [this, allocator] {
        if (opts.network_affinity >= 0)
            spead2::thread_pool::set_affinity(opts.network_affinity);
        for (std::size_t i = 0; i < chunking.n_chunks; i++)
            add_to_free(make_chunk(*allocator));
    });
    alloc_future.get();

    struct sigaction act = {}, old_act;
    act.sa_handler = signal_handler;
    act.sa_flags = SA_RESETHAND | SA_RESTART;
    int ret = sigaction(SIGINT, &act, &old_act);
    if (ret != 0)
        spead2::throw_errno("sigaction failed");

    std::future<void> collect_future;
    if (has_file)
        collect_future = std::async(std::launch::async, [this] { collect_thread(); });

    if (opts.network_affinity >= 0)
        spead2::thread_pool::set_affinity(opts.network_affinity);
    network_thread();
    ring.stop();

    /* Briefly sleep so that we can unsubscribe from the switch before we shut
     * down the QP. This makes it more likely that we can avoid incrementing
     * the dropped packets counter on the NIC.
     */
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (has_file)
        collect_future.get();
    // Restore SIGINT handler
    sigaction(SIGINT, &old_act, &act);
    time_point now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    std::cout << "\n\n" << packets << " packets captured (" << bytes << " bytes) in "
        << elapsed.count() << "s\n"
        << errors << " errors\n";
}

class capture : public capture_base
{
private:
    spead2::ibv_cq_t cq;
    spead2::ibv_qp_t qp;

    virtual void network_thread() override;
    virtual void post_chunk(chunk &c) override;
    virtual void init_record(chunk &c, std::size_t idx) override;

    static chunking_scheme sizes(const options &opts, const spead2::rdma_cm_id_t &cm_id);

public:
    capture(const options &opts);
};

capture::capture(const options &opts)
    : capture_base(opts, sizes)
{
    std::uint32_t n_slots = chunking.n_chunks * chunking.max_records;
#if SPEAD2_USE_IBV_EXP
    ibv_exp_cq_init_attr cq_attr = {};
    if (timestamp_support)
        cq_attr.flags |= IBV_EXP_CQ_TIMESTAMP;
    cq_attr.comp_mask = IBV_EXP_CQ_INIT_ATTR_FLAGS;
    cq = spead2::ibv_cq_t(cm_id, n_slots, nullptr, &cq_attr);
#else
    cq = spead2::ibv_cq_t(cm_id, n_slots, nullptr);
#endif

    ibv_qp_init_attr qp_attr = {};
    qp_attr.send_cq = cq.get();
    qp_attr.recv_cq = cq.get();
    qp_attr.qp_type = IBV_QPT_RAW_PACKET;
    qp_attr.cap.max_send_wr = 1;
    qp_attr.cap.max_recv_wr = n_slots;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    qp = spead2::ibv_qp_t(pd, &qp_attr);
    qp.modify(IBV_QPS_INIT, cm_id->port_num);
}

chunking_scheme capture::sizes(const options &opts, const spead2::rdma_cm_id_t &cm_id)
{
    constexpr std::size_t nominal_chunk_size = 2 * 1024 * 1024; // TODO: make tunable?
    std::size_t max_records = nominal_chunk_size / opts.snaplen;
    if (max_records == 0)
        max_records = 1;

    std::size_t chunk_size = max_records * opts.snaplen;
    std::size_t n_chunks = opts.net_buffer / chunk_size;
    if (n_chunks == 0)
        n_chunks = 1;

    ibv_device_attr attr = cm_id.query_device();
    unsigned int device_slots = std::min(attr.max_cqe, attr.max_qp_wr);
    unsigned int device_chunks = device_slots / max_records;
    if (attr.max_mr < device_chunks)
        device_chunks = attr.max_mr;

    bool reduced = false;
    if (device_slots < max_records)
    {
        max_records = device_slots;
        reduced = true;
    }
    if (device_slots < n_chunks * max_records)
    {
        n_chunks = device_slots / max_records;
        reduced = true;
    }
    if (reduced)
        spead2::log_warning("Reducing buffer to %d to accommodate device limits",
                            n_chunks * max_records * opts.snaplen);
    return {max_records, n_chunks, opts.snaplen * max_records};
}

void capture::network_thread()
{
    // Number of polls + packets between checking the time. Fetching the
    // time is expensive, so we don't want to do it too often.
    constexpr int GET_TIME_RATE = 10000;

    std::vector<boost::asio::ip::udp::endpoint> endpoints;
    for (const std::string &s : opts.endpoints)
        endpoints.push_back(make_endpoint(s));
    auto flows = spead2::create_flows(qp, endpoints, cm_id->port_num);
    qp.modify(IBV_QPS_RTR);
    joiner join(interface_address, endpoints);

    start_time = std::chrono::high_resolution_clock::now();
    last_report = start_time;
    const std::size_t max_records = chunking.max_records;
#if SPEAD2_USE_IBV_EXP
    std::unique_ptr<ibv_exp_wc[]> wc(new ibv_exp_wc[max_records]);
#else
    std::unique_ptr<ibv_wc[]> wc(new ibv_wc[max_records]);
#endif
    int until_get_time = GET_TIME_RATE;
    std::uint64_t remaining_count = opts.count;
    while (!stop.load() && remaining_count > 0)
    {
        chunk c = free_ring.pop();
        int expect = max_records;
        if (remaining_count < max_records)
            expect = remaining_count;
        while (!stop.load() && expect > 0)
        {
            int n = cq.poll(expect, wc.get());
            packets += n;
            for (int i = 0; i < n; i++)
            {
                if (wc[i].status != IBV_WC_SUCCESS)
                {
                    spead2::log_warning("failed WR %1%: %2% (vendor_err: %3%)",
                                        wc[i].wr_id, wc[i].status, wc[i].vendor_err);
                    errors++;
                    packets--;
                }
                else
                {
                    std::size_t idx = wc[i].wr_id;
                    assert(idx == c.n_records);
                    record_header &record = c.entries[idx].record;
                    record.incl_len = wc[i].byte_len;
                    record.orig_len = wc[i].byte_len;
#if SPEAD2_USE_IBV_EXP
                    if (timestamp_support && (wc[i].exp_wc_flags & IBV_EXP_WC_WITH_TIMESTAMP))
                    {
                        std::uint64_t ticks = wc[i].timestamp - timestamp_subtract;
                        std::uint64_t ns = std::uint64_t(ticks * hca_ns_per_clock);
                        ns += timestamp_add;
                        record.ts_sec = ns / 1000000000;
                        record.ts_nsec = ns % 1000000000;
                    }
                    else
                    {
                        record.ts_sec = 0;
                        record.ts_nsec = 0;
                    }
#endif
                    c.iov[2 * idx + 1].iov_len = wc[i].byte_len;
                    c.n_records++;
                    c.n_bytes += wc[i].byte_len + sizeof(record_header);
                    bytes += wc[i].byte_len;
                }
            }
            expect -= n;
            until_get_time -= n + 1;
            if (until_get_time <= 0)
            {
                until_get_time = GET_TIME_RATE;
                if (!opts.quiet)
                {
                    time_point now = std::chrono::high_resolution_clock::now();
                    if (now - last_report >= std::chrono::seconds(1))
                        report_rates(now);
                }
            }
            if (remaining_count != std::numeric_limits<std::uint64_t>::max())
                remaining_count -= n;
        }
        c.full = c.n_records == max_records;
        chunk_ready(std::move(c));
    }
}

void capture::post_chunk(chunk &c)
{
    qp.post_recv(&c.entries[0].wr);
}

void capture::init_record(chunk &c, std::size_t idx)
{
    std::uintptr_t ptr = (std::uintptr_t) c.storage.get();
    ptr += idx * opts.snaplen;

    c.entries[idx].wr.wr_id = idx;
    c.entries[idx].wr.next = (idx + 1 < chunking.max_records) ? &c.entries[idx + 1].wr : nullptr;
    c.entries[idx].wr.num_sge = 1;
    c.entries[idx].wr.sg_list = &c.entries[idx].sg;
    c.entries[idx].sg.addr = ptr;
    c.entries[idx].sg.length = opts.snaplen;
    c.entries[idx].sg.lkey = c.storage_mr->lkey;
    c.iov[2 * idx].iov_base = &c.entries[idx].record;
    c.iov[2 * idx].iov_len = sizeof(record_header);
    c.iov[2 * idx + 1].iov_base = (void *) ptr;
}

#if SPEAD2_USE_IBV_MPRQ

class capture_mprq : public capture_base
{
private:
    spead2::ibv_exp_res_domain_t res_domain;
    spead2::ibv_cq_t cq;
    spead2::ibv_exp_cq_family_v1_t cq_intf;
    spead2::ibv_exp_wq_t wq;
    spead2::ibv_exp_wq_family_t wq_intf;
    spead2::ibv_exp_rwq_ind_table_t rwq_ind_table;
    spead2::ibv_qp_t qp;

    virtual void network_thread() override;
    virtual void post_chunk(chunk &c) override;
    virtual void init_record(chunk &c, std::size_t idx) override;

    static chunking_scheme sizes(const options &opts, const spead2::rdma_cm_id_t &cm_id);

public:
    capture_mprq(const options &opts);
};

static int log2i(std::size_t value)
{
    int x = 0;
    while ((value >> x) != 1)
        x++;
    assert(value == std::size_t(1) << x);
    return x;
}

capture_mprq::capture_mprq(const options &opts)
    : capture_base(opts, sizes)
{
    const std::size_t max_cqe = chunking.max_records * chunking.n_chunks;

    ibv_exp_res_domain_init_attr res_domain_attr = {};
    res_domain_attr.comp_mask = IBV_EXP_RES_DOMAIN_THREAD_MODEL | IBV_EXP_RES_DOMAIN_MSG_MODEL;
    res_domain_attr.thread_model = IBV_EXP_THREAD_UNSAFE;
    res_domain_attr.msg_model = IBV_EXP_MSG_HIGH_BW;
    res_domain = spead2::ibv_exp_res_domain_t(cm_id, &res_domain_attr);

    ibv_exp_cq_init_attr cq_attr = {};
    if (timestamp_support)
        cq_attr.flags |= IBV_EXP_CQ_TIMESTAMP;
    cq_attr.res_domain = res_domain.get();
    cq_attr.comp_mask = IBV_EXP_CQ_INIT_ATTR_FLAGS | IBV_EXP_CQ_INIT_ATTR_RES_DOMAIN;
    cq = spead2::ibv_cq_t(cm_id, max_cqe, nullptr, &cq_attr);
    cq_intf = spead2::ibv_exp_cq_family_v1_t(cm_id, cq);

    ibv_exp_wq_init_attr wq_attr = {};
    std::size_t stride = chunking.chunk_size / chunking.max_records;
    wq_attr.mp_rq.single_stride_log_num_of_bytes = log2i(stride);
    wq_attr.mp_rq.single_wqe_log_num_of_strides = log2i(chunking.max_records);
    wq_attr.mp_rq.use_shift = IBV_EXP_MP_RQ_NO_SHIFT;
    wq_attr.wq_type = IBV_EXP_WQT_RQ;
    wq_attr.max_recv_wr = chunking.n_chunks;
    wq_attr.max_recv_sge = 1;
    wq_attr.pd = pd.get();
    wq_attr.cq = cq.get();
    wq_attr.res_domain = res_domain.get();
    wq_attr.comp_mask = IBV_EXP_CREATE_WQ_MP_RQ | IBV_EXP_CREATE_WQ_RES_DOMAIN;
    wq = spead2::ibv_exp_wq_t(cm_id, &wq_attr);
    wq_intf = spead2::ibv_exp_wq_family_t(cm_id, wq);

    rwq_ind_table = spead2::create_rwq_ind_table(cm_id, pd, wq);

    /* ConnectX-5 only seems to work with Toeplitz hashing. This code seems to
     * work, but I don't really know what I'm doing so it might be horrible.
     */
    uint8_t toeplitz_key[40] = {};
    ibv_exp_rx_hash_conf hash_conf = {};
    hash_conf.rx_hash_function = IBV_EXP_RX_HASH_FUNC_TOEPLITZ;
    hash_conf.rx_hash_key_len = sizeof(toeplitz_key);
    hash_conf.rx_hash_key = toeplitz_key;
    hash_conf.rx_hash_fields_mask = 0;
    hash_conf.rwq_ind_tbl = rwq_ind_table.get();

    ibv_exp_qp_init_attr qp_attr = {};
    qp_attr.qp_type = IBV_QPT_RAW_PACKET;
    qp_attr.pd = pd.get();
    qp_attr.res_domain = res_domain.get();
    qp_attr.rx_hash_conf = &hash_conf;
    qp_attr.port_num = cm_id->port_num;
    qp_attr.comp_mask = IBV_EXP_QP_INIT_ATTR_PD | IBV_EXP_QP_INIT_ATTR_RX_HASH
        | IBV_EXP_QP_INIT_ATTR_PORT | IBV_EXP_QP_INIT_ATTR_RES_DOMAIN;
    qp = spead2::ibv_qp_t(cm_id, &qp_attr);

    wq.modify(IBV_EXP_WQS_RDY);
}

static int clamp(int x, int low, int high)
{
    return std::min(std::max(x, low), high);
}

chunking_scheme capture_mprq::sizes(const options &opts, const spead2::rdma_cm_id_t &cm_id)
{
    ibv_exp_device_attr attr = cm_id.exp_query_device();
    if (!(attr.comp_mask & IBV_EXP_DEVICE_ATTR_MP_RQ)
        || !(attr.mp_rq_caps.supported_qps & IBV_EXP_MP_RQ_SUP_TYPE_WQ_RQ))
        throw std::system_error(std::make_error_code(std::errc::not_supported),
                                "device does not support multi-packet receive queues");

    /* TODO: adapt these to the requested buffer size e.g. if a very large
     * buffer is requested, might need to increase the stride size to avoid
     * running out of CQEs.
     */
    std::size_t log_stride_bytes =
        clamp(6,
              attr.mp_rq_caps.min_single_stride_log_num_of_bytes,
              attr.mp_rq_caps.max_single_stride_log_num_of_bytes);   // 64 bytes
    std::size_t log_strides_per_chunk =
        clamp(21 - log_stride_bytes,
              attr.mp_rq_caps.min_single_wqe_log_num_of_strides,
              attr.mp_rq_caps.max_single_wqe_log_num_of_strides);    // 2MB chunks
    std::size_t max_records = 1 << log_strides_per_chunk;
    std::size_t chunk_size = max_records << log_stride_bytes;
    std::size_t n_chunks = opts.net_buffer / chunk_size;
    if (n_chunks == 0)
        n_chunks = 1;

    unsigned int device_chunks = std::min(attr.max_qp_wr, attr.max_mr);

    bool reduced = false;
    if (device_chunks < n_chunks)
    {
        n_chunks = device_chunks;
        reduced = true;
    }

    if (reduced)
        spead2::log_warning("Reducing buffer to %d to accommodate device limits",
                            n_chunks * chunk_size);
    return {max_records, n_chunks, chunk_size};
}

void capture_mprq::network_thread()
{
    // Number of polls + packets between checking the time. Fetching the
    // time is expensive, so we don't want to do it too often.
    constexpr int GET_TIME_RATE = 10000;

    std::vector<boost::asio::ip::udp::endpoint> endpoints;
    for (const std::string &s : opts.endpoints)
        endpoints.push_back(make_endpoint(s));
    auto flows = spead2::create_flows(qp, endpoints, cm_id->port_num);
    joiner join(interface_address, endpoints);

    start_time = std::chrono::high_resolution_clock::now();
    last_report = start_time;
    const std::size_t max_records = chunking.max_records;
    int until_get_time = GET_TIME_RATE;
    std::uint64_t remaining_count = opts.count;
    while (!stop.load() && remaining_count > 0)
    {
        chunk c = free_ring.pop();
        while (!stop.load() && remaining_count > 0 && !c.full)
        {
            std::uint32_t offset;
            std::uint32_t flags;
            std::uint64_t timestamp;
            std::int32_t len = cq_intf->poll_length_flags_mp_rq_ts(
                cq.get(), &offset, &flags, &timestamp);
            if (len < 0)
            {
                // Error condition.
                ibv_wc wc;
                ibv_poll_cq(cq.get(), 1, &wc);
                spead2::log_warning("failed WR %1%: %2% (vendor_err: %3%)",
                                    wc.wr_id, wc.status, wc.vendor_err);
                errors++;
            }
            else if (len > 0)
            {
                packets++;
                if (remaining_count != std::numeric_limits<std::uint64_t>::max())
                    remaining_count--;
                std::size_t idx = c.n_records;
                record_header &record = c.entries[idx].record;
                record.incl_len = (len <= opts.snaplen) ? len : opts.snaplen;
                record.orig_len = len;
                if (timestamp_support && (flags & IBV_EXP_CQ_RX_WITH_TIMESTAMP))
                {
                    std::uint64_t ticks = timestamp - timestamp_subtract;
                    std::uint64_t ns = std::uint64_t(ticks * hca_ns_per_clock);
                    ns += timestamp_add;
                    record.ts_sec = ns / 1000000000;
                    record.ts_nsec = ns % 1000000000;
                }
                else
                {
                    record.ts_sec = 0;
                    record.ts_nsec = 0;
                }
                c.iov[2 * idx + 1].iov_base = &c.storage[offset];
                c.iov[2 * idx + 1].iov_len = record.incl_len;
                c.n_records++;
                c.n_bytes += len + sizeof(record_header);
                bytes += len;
            }
            until_get_time--;
            if (until_get_time <= 0)
            {
                // TODO: unify this code with non-mprq
                until_get_time = GET_TIME_RATE;
                if (!opts.quiet)
                {
                    time_point now = std::chrono::high_resolution_clock::now();
                    if (now - last_report >= std::chrono::seconds(1))
                        report_rates(now);
                }
            }
            if (flags & IBV_EXP_CQ_RX_MULTI_PACKET_LAST_V1)
                c.full = true;
        }
        chunk_ready(std::move(c));
    }
}

void capture_mprq::post_chunk(chunk &c)
{
    ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t) c.storage.get();
    sge.length = chunking.chunk_size;
    sge.lkey = c.storage_mr->lkey;
    int status = wq_intf->recv_burst(wq.get(), &sge, 1);
    if (status != 0)
        spead2::throw_errno("recv_burst failed", status);
}

void capture_mprq::init_record(chunk &c, std::size_t idx)
{
    c.iov[2 * idx].iov_base = &c.entries[idx].record;
    c.iov[2 * idx].iov_len = sizeof(record_header);
}

#endif // SPEAD2_USE_IBV_MPRQ

int main(int argc, const char **argv)
{
    try
    {
        spead2::set_log_function(log_function);
        options opts = parse_args(argc, argv);
        std::unique_ptr<capture_base> cap;

#if SPEAD2_USE_IBV_MPRQ
        try
        {
            cap.reset(new capture_mprq(opts));
        }
        catch (std::system_error &e)
        {
            if (e.code() != std::errc::not_supported)
                throw;
        }
#endif
        if (!cap)
            cap.reset(new capture(opts));
        cap->run();
    }
    catch (std::runtime_error &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
