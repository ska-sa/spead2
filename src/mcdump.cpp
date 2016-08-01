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
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <atomic>
#include <unistd.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <signal.h>

namespace po = boost::program_options;

struct options
{
    std::string multicast_group;
    std::uint16_t port;
    std::string interface;
    std::string filename;
    int snaplen = 9230;
    std::size_t buffer = 128 * 1024 * 1024;
};

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: mcdump [options] -i <iface-addr> <group> <port> <filename>\n";
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
        ("buffer", make_opt(opts.buffer), "Maximum memory for buffering")
        ("help,h", "Show help text")
    ;

    hidden.add_options()
        ("group", make_opt_nodefault(opts.multicast_group), "multicast-group to receive")
        ("port", make_opt_nodefault(opts.port), "UDP port number")
        ("filename", make_opt_nodefault(opts.filename), "output filename")
    ;
    all.add(desc);
    all.add(hidden);

    po::positional_options_description positional;
    positional.add("group", 1);
    positional.add("port", 1);
    positional.add("filename", 1);
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
        if (!vm.count("filename"))
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

typedef spead2::ringbuffer<chunk> ringbuffer;
static std::atomic<bool> stop{false};

static void signal_handler(int)
{
    stop = true;
}

class capture
{
private:
    const options opts;
    std::size_t max_records;
    int fd = -1;
    ringbuffer ring;
    ringbuffer free_ring;
    spead2::rdma_event_channel_t event_channel;
    spead2::rdma_cm_id_t cm_id;
    spead2::ibv_qp_t qp;
    spead2::ibv_pd_t pd;
    spead2::ibv_cq_t cq;
    spead2::ibv_flow_t flow;
    std::uint64_t errors = 0;

    chunk make_chunk(spead2::memory_allocator &allocator);
    void add_to_free(chunk &&c);

    void disk_thread();
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

void capture::disk_thread()
{
    std::uint32_t iov_max = sysconf(_SC_IOV_MAX);
    while (true)
    {
        try
        {
            chunk c = ring.pop();
            std::uint32_t n_iov = 2 * c.n_records;
            for (std::uint32_t offset = 0; offset < n_iov; offset += iov_max)
            {
                std::uint32_t n = std::min(n_iov - offset, iov_max);
                ssize_t ret = writev(fd, c.iov.get() + offset, n);
                if (ret != c.n_bytes)
                    spead2::throw_errno("writev failed");
            }
            add_to_free(std::move(c));
        }
        catch (spead2::ringbuffer_stopped)
        {
            free_ring.stop();
            int ret = close(fd);
            if (ret < 0)
                spead2::throw_errno("close failed");
            return;
        }
    }
}

void capture::network_thread()
{
    std::unique_ptr<ibv_wc[]> wc(new ibv_wc[max_records]);
    while (!stop.load())
    {
        chunk c = free_ring.pop();
        int expect = max_records;
        while (!stop.load() && expect > 0)
        {
            int n = cq.poll(expect, wc.get());
            for (int i = 0; i < n; i++)
            {
                if (wc[i].status != IBV_WC_SUCCESS)
                {
                    spead2::log_warning("failed WR %1%: %2% (vendor_err: %3%)",
                                        wc[i].wr_id, wc[i].status, wc[i].vendor_err);
                    errors++;
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
    std::size_t n_chunks = opts.buffer / chunk_size;
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

void capture::run()
{
    using boost::asio::ip::udp;

    fd = open(opts.filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0)
        spead2::throw_errno("open failed");
    file_header header;
    header.snaplen = opts.snaplen;
    if (write(fd, &header, sizeof(header)) < sizeof(header))
        spead2::throw_errno("write failed");

    boost::asio::io_service io_service;
    udp::endpoint endpoint(
        boost::asio::ip::address_v4::from_string(opts.multicast_group),
        opts.port);
    boost::asio::ip::address_v4 interface_address =
        boost::asio::ip::address_v4::from_string(opts.interface);
    if (!endpoint.address().is_multicast())
        throw std::runtime_error("Address " + opts.multicast_group + " is not a multicast address");
    udp::socket join_socket(io_service, endpoint.protocol());
    join_socket.set_option(boost::asio::socket_base::reuse_address(true));
    join_socket.set_option(boost::asio::ip::multicast::join_group(
        endpoint.address().to_v4(), interface_address));

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
    flow = create_flow(qp, endpoint, cm_id->port_num);

    std::shared_ptr<spead2::mmap_allocator> allocator =
        std::make_shared<spead2::mmap_allocator>(0, true);
    for (std::size_t i = 0; i < n_chunks; i++)
        add_to_free(make_chunk(*allocator));
    qp.modify(IBV_QPS_RTR);

    struct sigaction act = {}, old_act;
    act.sa_handler = signal_handler;
    act.sa_flags = SA_RESETHAND | SA_RESTART;
    int ret = sigaction(SIGINT, &act, &old_act);
    if (ret != 0)
        spead2::throw_errno("sigaction failed");

    std::future<void> network_future = std::async(std::launch::async, [this] { network_thread(); });
    disk_thread();
    network_future.get();
    // Restore SIGINT handler
    sigaction(SIGINT, &old_act, &act);
}

int main(int argc, const char **argv)
{
    try
    {
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
