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

/**
 * @file
 *
 * Utility program to dump raw multicast packets, using ibverbs. It works with
 * any multicast UDP data, not just SPEAD.
 */

#include <spead2/common_ibv.h>
#include <spead2/common_raw_packet.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_logging.h>
#include <spead2/common_memory_pool.h>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <utility>
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
#ifdef O_DIRECT
    bool direct = false;
#endif
};

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: mcdump [options] -i <iface-addr> <filename> <group>:<port>...\n";
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
    std::uint32_t ts_usec = 0;
    std::uint32_t incl_len;
    std::uint32_t orig_len;
};

struct chunk_entry
{
    ibv_recv_wr wr;
    ibv_sge sg;
    record_header record;
};

struct chunk
{
    std::uint32_t n_records;
    std::size_t n_bytes;
    std::unique_ptr<chunk_entry[]> entries;
    std::unique_ptr<iovec[]> iov;
    spead2::memory_pool::pointer storage;
    spead2::ibv_mr_t records_mr, storage_mr;
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
        catch (spead2::ringbuffer_stopped)
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
            catch (spead2::ringbuffer_stopped)
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

class capture
{
private:
    typedef spead2::ringbuffer<chunk> ringbuffer;

    const options opts;
    std::size_t max_records;
    std::unique_ptr<writer> w;
    ringbuffer ring;
    ringbuffer free_ring;
    spead2::rdma_event_channel_t event_channel;
    spead2::rdma_cm_id_t cm_id;
    spead2::ibv_qp_t qp;
    spead2::ibv_pd_t pd;
    spead2::ibv_cq_t cq;
    std::vector<spead2::ibv_flow_t> flows;
    std::uint64_t errors = 0;
    std::uint64_t packets = 0;
    std::uint64_t bytes = 0;

    chunk make_chunk(spead2::memory_allocator &allocator);
    void add_to_free(chunk &&c);

    void collect_thread();
    void network_thread();

public:
    capture(const options &opts);
    void run();
};

chunk capture::make_chunk(spead2::memory_allocator &allocator)
{
    chunk c;
    c.n_records = 0;
    c.n_bytes = 0;
    c.entries.reset(new chunk_entry[max_records]);
    c.iov.reset(new iovec[2 * max_records]);
    c.storage = allocator.allocate(opts.snaplen * max_records, nullptr);
    c.storage_mr = spead2::ibv_mr_t(pd, c.storage.get(), opts.snaplen * max_records, IBV_ACCESS_LOCAL_WRITE);
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

void capture::add_to_free(chunk &&c)
{
    c.n_records = 0;
    c.n_bytes = 0;
    qp.post_recv(&c.entries[0].wr);
    free_ring.push(std::move(c));
}

void capture::collect_thread()
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

                /* Only post a new receive if the chunk was full. It if was
                 * not full, then this was the last chunk, and we're about to
                 * get a stop. Some of the work requests are already in the
                 * queue, so posting them again is asking for trouble.
                 *
                 * If the chunk was not full, we can't just free it, because
                 * the QP might still be receiving data and writing it to the
                 * chunk. So we push it back onto the ring without posting a
                 * new receive, just to keep it live.
                 */
                if (c.n_records == max_records)
                    add_to_free(std::move(c));
                else
                    free_ring.push(std::move(c));
            }
            catch (spead2::ringbuffer_stopped)
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

void capture::network_thread()
{
    if (opts.network_affinity >= 0)
        spead2::thread_pool::set_affinity(opts.network_affinity);
    std::unique_ptr<ibv_wc[]> wc(new ibv_wc[max_records]);
    while (!stop.load())
    {
        chunk c = free_ring.pop();
        int expect = max_records;
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
                    c.entries[idx].record.incl_len = wc[i].byte_len;
                    c.entries[idx].record.orig_len = wc[i].byte_len;
                    c.iov[2 * idx + 1].iov_len = wc[i].byte_len;
                    c.n_records++;
                    c.n_bytes += wc[i].byte_len + sizeof(record_header);
                    bytes += wc[i].byte_len;
                }
            }
            expect -= n;
        }
        ring.push(std::move(c));
    }
    ring.stop();
}

static spead2::ibv_flow_t create_flow(
    const spead2::ibv_qp_t &qp, const boost::asio::ip::udp::endpoint &endpoint, int port_num)
{
    struct
    {
        ibv_flow_attr attr;
        ibv_flow_spec_eth eth;
        ibv_flow_spec_ipv4 ip;
        ibv_flow_spec_tcp_udp udp;
    } __attribute__((packed)) flow_rule;
    memset(&flow_rule, 0, sizeof(flow_rule));

    flow_rule.attr.type = IBV_FLOW_ATTR_NORMAL;
    flow_rule.attr.priority = 0;
    flow_rule.attr.size = sizeof(flow_rule);
    flow_rule.attr.num_of_specs = 3;
    flow_rule.attr.port = port_num;

    flow_rule.eth.type = IBV_FLOW_SPEC_ETH;
    flow_rule.eth.size = sizeof(flow_rule.eth);
    spead2::mac_address dst_mac = spead2::multicast_mac(endpoint.address());
    std::memcpy(&flow_rule.eth.val.dst_mac, &dst_mac, sizeof(dst_mac));
    // Set all 1's mask
    std::memset(&flow_rule.eth.mask.dst_mac, 0xFF, sizeof(flow_rule.eth.mask.dst_mac));

    flow_rule.ip.type = IBV_FLOW_SPEC_IPV4;
    flow_rule.ip.size = sizeof(flow_rule.ip);
    auto bytes = endpoint.address().to_v4().to_bytes(); // big-endian address
    std::memcpy(&flow_rule.ip.val.dst_ip, &bytes, sizeof(bytes));
    std::memset(&flow_rule.ip.mask.dst_ip, 0xFF, sizeof(flow_rule.ip.mask.dst_ip));

    flow_rule.udp.type = IBV_FLOW_SPEC_UDP;
    flow_rule.udp.size = sizeof(flow_rule.udp);
    flow_rule.udp.val.dst_port = htobe16(endpoint.port());
    flow_rule.udp.mask.dst_port = 0xFFFF;

    return spead2::ibv_flow_t(qp, &flow_rule.attr);
}

static spead2::ibv_qp_t create_qp(
    const spead2::ibv_pd_t &pd, const spead2::ibv_cq_t &cq, std::uint32_t n_slots)
{
    ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = cq.get();
    attr.recv_cq = cq.get();
    attr.qp_type = IBV_QPT_RAW_PACKET;
    attr.cap.max_send_wr = 1;
    attr.cap.max_recv_wr = n_slots;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    return spead2::ibv_qp_t(pd, &attr);
}

// Returns number of records per chunk and number of chunks
static std::pair<std::size_t, std::size_t> sizes(const options &opts)
{
    constexpr std::size_t nominal_chunk_size = 2 * 1024 * 1024; // TODO: make tunable?
    std::size_t max_records = nominal_chunk_size / opts.snaplen;
    if (max_records == 0)
        max_records = 1;
    std::size_t chunk_size = max_records * opts.snaplen;
    std::size_t n_chunks = opts.net_buffer / chunk_size;
    if (n_chunks == 0)
        n_chunks++;
    return {max_records, n_chunks};
}

capture::capture(const options &opts)
    : opts(opts), max_records(sizes(opts).first),
    ring(sizes(opts).second),
    free_ring(sizes(opts).second)
{
}

static boost::asio::ip::udp::endpoint make_endpoint(const std::string &s)
{
    // Use rfind rather than find because IPv6 addresses contain :'s
    auto pos = s.rfind(':');
    try
    {
        boost::asio::ip::address_v4 addr = boost::asio::ip::address_v4::from_string(s.substr(0, pos));
        if (!addr.is_multicast())
            throw std::runtime_error("Address " + s.substr(0, pos) + " is not a multicast address");
        std::uint16_t port = boost::lexical_cast<std::uint16_t>(s.substr(pos + 1));
        return boost::asio::ip::udp::endpoint(addr, port);
    }
    catch (boost::bad_lexical_cast)
    {
        throw std::runtime_error("Invalid port number " + s.substr(pos + 1));
    }
}

void capture::run()
{
    using boost::asio::ip::udp;

    std::shared_ptr<spead2::mmap_allocator> allocator =
        std::make_shared<spead2::mmap_allocator>(0, true);
    int fd_flags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_DIRECT
    if (opts.direct)
        fd_flags |= O_DIRECT;
#endif
    int fd = open(opts.filename.c_str(), fd_flags, 0666);
    if (fd < 0)
        spead2::throw_errno("open failed");
    w.reset(new writer(opts, fd, *allocator));

    boost::asio::io_service io_service;
    std::vector<udp::endpoint> endpoints;
    for (const std::string &s : opts.endpoints)
        endpoints.push_back(make_endpoint(s));
    boost::asio::ip::address_v4 interface_address =
        boost::asio::ip::address_v4::from_string(opts.interface);

    std::size_t n_chunks = sizes(opts).second;
    if (std::numeric_limits<std::uint32_t>::max() / max_records <= n_chunks)
        throw std::runtime_error("Too many buffered packets");
    std::uint32_t n_slots = n_chunks * max_records;
    cm_id = spead2::rdma_cm_id_t(event_channel, nullptr, RDMA_PS_UDP);
    cm_id.bind_addr(interface_address);
    cq = spead2::ibv_cq_t(cm_id, n_slots, nullptr);
    pd = spead2::ibv_pd_t(cm_id);
    qp = create_qp(pd, cq, n_slots);
    qp.modify(IBV_QPS_INIT, cm_id->port_num);
    for (const udp::endpoint &endpoint : endpoints)
        flows.push_back(create_flow(qp, endpoint, cm_id->port_num));

    for (std::size_t i = 0; i < n_chunks; i++)
        add_to_free(make_chunk(*allocator));
    qp.modify(IBV_QPS_RTR);

    struct sigaction act = {}, old_act;
    act.sa_handler = signal_handler;
    act.sa_flags = SA_RESETHAND | SA_RESTART;
    int ret = sigaction(SIGINT, &act, &old_act);
    if (ret != 0)
        spead2::throw_errno("sigaction failed");

    std::future<void> collect_future = std::async(std::launch::async, [this] { collect_thread(); });

    udp::socket join_socket(io_service, endpoints[0].protocol());
    join_socket.set_option(boost::asio::socket_base::reuse_address(true));
    for (const udp::endpoint &endpoint : endpoints)
        join_socket.set_option(boost::asio::ip::multicast::join_group(
            endpoint.address().to_v4(), interface_address));

    network_thread();
    /* Close socket then briefly sleep so that we can unsubscribe from the
     * switch before we shut down the QP. This makes it more likely that we
     * can avoid incrementing the dropped packets counter on the NIC.
     */
    join_socket.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    collect_future.get();
    // Restore SIGINT handler
    sigaction(SIGINT, &old_act, &act);
    std::cout << "\n\n" << packets << " packets captured (" << bytes << " bytes)\n"
        << errors << " errors\n";
}

int main(int argc, const char **argv)
{
    try
    {
        spead2::set_log_function(log_function);
        options opts = parse_args(argc, argv);
        capture cap(opts);
        cap.run();
    }
    catch (std::runtime_error &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
