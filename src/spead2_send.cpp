/* Copyright 2016, 2017, 2019-2020 National Research Foundation (SARAO)
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
#include <random>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <spead2/common_thread_pool.h>
#include <spead2/common_semaphore.h>
#include <spead2/common_endian.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/send_stream.h>
#include <spead2/send_udp.h>
#include <spead2/send_tcp.h>
#include <spead2/common_features.h>
#if SPEAD2_USE_IBV
# include <spead2/send_udp_ibv.h>
#endif
#include "spead2_cmdline.h"

namespace po = boost::program_options;
namespace asio = boost::asio;
using boost::asio::ip::udp;
using boost::asio::ip::tcp;

struct options
{
    spead2::protocol_options protocol;
    spead2::send::sender_options sender;
    std::size_t heap_size = 4194304;
    std::size_t items = 1;
    std::int64_t heaps = -1;
    bool verify = false;
    std::vector<std::string> dest;
};

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: spead2_send [options] <host>:<port> [<host>:<port> ...]\n";
    o << desc;
}

static options parse_args(int argc, const char **argv)
{
    options opts;
    po::options_description desc, hidden, all;
    desc.add_options()
        ("heap-size", spead2::make_value_semantic(&opts.heap_size), "Payload size for heap")
        ("items", spead2::make_value_semantic(&opts.items), "Number of items per heap")
        ("heaps", spead2::make_value_semantic(&opts.heaps), "Number of data heaps to send (-1=infinite)")
        ("verify", spead2::make_value_semantic(&opts.verify), "Insert payload values that receiver can verify")
    ;
    spead2::option_adder adder(desc);
    opts.protocol.enumerate(adder);
    opts.sender.enumerate(adder);
    hidden.add_options()
        ("destination", spead2::make_value_semantic(&opts.dest), "Destination host:port")
    ;
    desc.add_options()
        ("help,h", "Show help text");
    all.add(desc);
    all.add(hidden);

    po::positional_options_description positional;
    positional.add("destination", -1);
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
        if (!vm.count("destination"))
            throw po::error("at least one destination is required");
        if (opts.dest.size() > 1 && opts.protocol.tcp)
            throw po::error("only one destination is supported with TCP");
        opts.sender.notify(opts.protocol);
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
    const std::size_t max_heaps;
    const std::size_t n_substreams;
    const std::int64_t n_heaps;
    const spead2::flavour flavour;
    const std::size_t elements;                              // number of element_t's per item
    const bool verify;

    std::vector<spead2::send::heap> first_heaps;             // have descriptors
    std::vector<spead2::send::heap> heaps;
    spead2::send::heap last_heap;                            // has end-of-stream marker
    typedef std::uint32_t element_t;
    spead2::memory_allocator::pointer storage;               // data for all heaps
    std::vector<std::vector<element_t *>> pointers;          // start of data per item per heap
    std::minstd_rand generator;

    std::uint64_t bytes_transferred = 0;
    boost::system::error_code error;
    spead2::semaphore done_sem{0};

    const spead2::send::heap &get_heap(std::uint64_t idx) noexcept;

    void callback(spead2::send::stream &stream,
                  std::uint64_t idx,
                  const boost::system::error_code &ec,
                  std::size_t bytes_transferred);

public:
    explicit sender(const options &opts);
    std::uint64_t run(spead2::send::stream &stream);

    std::vector<std::pair<const void *, std::size_t>> memory_regions() const;
};

sender::sender(const options &opts)
    : max_heaps((opts.heaps < 0
                 || std::uint64_t(opts.heaps) + opts.dest.size() > opts.sender.max_heaps)
                ? opts.sender.max_heaps : opts.heaps + opts.dest.size()),
    n_substreams(opts.dest.size()),
    n_heaps(opts.heaps),
    elements(opts.heap_size / (opts.items * sizeof(element_t))),
    verify(opts.verify),
    last_heap(opts.sender.make_flavour(opts.protocol))
{
    first_heaps.reserve(max_heaps);
    heaps.reserve(max_heaps);
    for (std::size_t i = 0; i < max_heaps; i++)
    {
        first_heaps.emplace_back(flavour);
        heaps.emplace_back(flavour);
    }

    const std::size_t item_size = elements * sizeof(element_t);
    const std::size_t heap_size = item_size * opts.items;
    if (heap_size != opts.heap_size)
    {
        std::cerr << "Heap size is not an exact multiple: using " << heap_size << " instead of " << opts.heap_size << '\n';
    }

    auto allocator = std::make_shared<spead2::mmap_allocator>(0, true);
    storage = allocator->allocate(heap_size * max_heaps, nullptr);

    for (std::size_t i = 0; i < opts.items; i++)
    {
        spead2::descriptor d;
        d.id = 0x1000 + i;
        std::ostringstream sstr;
        sstr << "Test item " << i;
        d.name = sstr.str();
        d.description = "A test item with arbitrary value";
        sstr.str("");
        sstr << "{'shape': (" << elements << ",), 'fortran_order': False, 'descr': '>u4'}";
        d.numpy_header = sstr.str();
        for (std::size_t j = 0; j < max_heaps; j++)
            first_heaps[j].add_descriptor(d);
    }

    pointers.resize(max_heaps);
    for (std::size_t i = 0; i < max_heaps; i++)
    {
        pointers[i].resize(opts.items);
        for (std::size_t j = 0; j < opts.items; j++)
        {
            pointers[i][j] = reinterpret_cast<element_t *>(
                storage.get() + i * heap_size + j * item_size);
            first_heaps[i].add_item(0x1000 + j, pointers[i][j], item_size, true);
            heaps[i].add_item(0x1000 + j, pointers[i][j], item_size, true);
        }
    }
    last_heap.add_end();
}

std::vector<std::pair<const void *, std::size_t>> sender::memory_regions() const
{
    return {{storage.get(), max_heaps * pointers[0].size() * elements * sizeof(element_t)}};
}

const spead2::send::heap &sender::get_heap(std::uint64_t idx) noexcept
{
    spead2::send::heap *heap;
    if (idx < n_substreams)
        heap = &first_heaps[idx % max_heaps];
    else if (n_heaps >= 0 && idx >= std::uint64_t(n_heaps))
        return last_heap;
    else
        heap = &heaps[idx % max_heaps];

    if (verify)
    {
        const std::vector<element_t *> &ptrs = pointers[idx % max_heaps];
        // Fill in random values to be checked by the receiver
        for (std::size_t i = 0; i < ptrs.size(); i++)
            for (std::size_t j = 0; j < elements; j++)
                ptrs[i][j] = spead2::htobe(std::uint32_t(generator()));
    }
    return *heap;
}

void sender::callback(spead2::send::stream &stream,
                      std::uint64_t idx,
                      const boost::system::error_code &ec,
                      std::size_t bytes_transferred)
{
    this->bytes_transferred += bytes_transferred;
    if (ec && !error)
        error = ec;
    if (error)
    {
        done_sem.put();
        return;
    }

    idx += max_heaps;
    if (n_heaps == -1 || idx < std::uint64_t(n_heaps) + n_substreams)
    {
        stream.async_send_heap(
            get_heap(idx),
            [this, &stream, idx] (const boost::system::error_code &ec, std::size_t bytes_transferred) {
                callback(stream, idx, ec, bytes_transferred);
            }, -1, idx % n_substreams);
    }
    else
        done_sem.put();
}

std::uint64_t sender::run(spead2::send::stream &stream)
{
    bytes_transferred = 0;
    error = boost::system::error_code();
    /* Send the initial heaps from the worker thread. This ensures that no
     * callbacks can happen until the initial heaps are all sent, which would
     * otherwise lead to heaps being queued out of order.
     */
    stream.get_io_service().post([this, &stream] {
        for (int i = 0; i < max_heaps; i++)
            stream.async_send_heap(
                get_heap(i),
                [this, &stream, i] (const boost::system::error_code &ec, std::size_t bytes_transferred) {
                    callback(stream, i, ec, bytes_transferred);
                }, -1, i % n_substreams);
    });
    for (int i = 0; i < max_heaps; i++)
        semaphore_get(done_sem);
    if (error)
        throw boost::system::system_error(error);
    return bytes_transferred;
}

} // anonymous namespace

static int run(spead2::send::stream &stream, sender &s)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::uint64_t sent_bytes = s.run(stream);
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop_time - start_time;
    double elapsed_s = elapsed.count();
    std::cout
        << "Sent " << sent_bytes << " bytes in " << elapsed_s << " seconds, "
        << sent_bytes * 8.0e-9 / elapsed_s << " Gb/s\n";
    return 0;
}

template <typename Proto>
static std::vector<boost::asio::ip::basic_endpoint<Proto>> get_endpoints(
    boost::asio::io_service &io_service, const options &opts)
{
    typedef boost::asio::ip::basic_resolver<Proto> resolver_type;
    resolver_type resolver(io_service);
    std::vector<boost::asio::ip::basic_endpoint<Proto>> ans;
    ans.reserve(opts.dest.size());
    for (const std::string &dest : opts.dest)
    {
        auto colon = dest.rfind(':');
        if (colon == std::string::npos)
            throw std::runtime_error("Destination '" + dest + "' does not have the format host:port");
        typename resolver_type::query query(dest.substr(0, colon), dest.substr(colon + 1));
        ans.push_back(*resolver.resolve(query));
    }
    return ans;
}

int main(int argc, const char **argv)
{
    options opts = parse_args(argc, argv);

    sender s(opts);

    spead2::thread_pool thread_pool(1);
    std::unique_ptr<spead2::send::stream> stream =
        opts.sender.make_stream(thread_pool.get_io_service(), opts.protocol,
                                opts.dest, s.memory_regions());
    return run(*stream, s);
}
