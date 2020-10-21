/* Copyright 2015, 2017, 2020 National Research Foundation (SARAO)
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
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <stdexcept>
#include <cstdint>
#include <unistd.h>
#include <sys/socket.h>
#include <boost/optional.hpp>
#include <spead2/recv_udp.h>
#include <spead2/recv_udp_ibv.h>
#include <spead2/recv_udp_pcap.h>
#include <spead2/recv_tcp.h>
#include <spead2/recv_mem.h>
#include <spead2/recv_inproc.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_ring_stream.h>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_heap.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/py_common.h>

namespace py = pybind11;

namespace spead2
{
namespace recv
{

/* True if both arguments are plain function pointers and point to the same function */
template<typename R, typename... Args>
static bool equal_functions(const std::function<R(Args...)> &a,
                            const std::function<R(Args...)> &b)
{
    using ptr = R (*)(Args...);
    const ptr *x = a.template target<ptr>();
    const ptr *y = b.template target<ptr>();
    return x && y && *x == *y;
}

/**
 * Wraps @ref item to provide safe memory management. The item references
 * memory inside the heap, so it needs to hold a reference to that
 * heap, as do any memoryviews created on the value.
 */
class item_wrapper : public item
{
private:
    /// Python object containing a @ref heap
    py::object owning_heap;

public:
    item_wrapper() = default;
    item_wrapper(const item &it, py::object owning_heap)
        : item(it), owning_heap(std::move(owning_heap)) {}

    /**
     * Obtain the raw value using Python buffer protocol.
     */
    py::buffer_info get_value() const
    {
        return py::buffer_info(
            reinterpret_cast<void *>(ptr),
            1,      // size of individual elements
            py::format_descriptor<std::uint8_t>::format(),
            length);
    }
};

/**
 * Extends mem_reader to obtain data using the Python buffer protocol.
 * It steals the provided buffer view; it is not passed by rvalue reference
 * because it cannot be perfectly forwarded.
 */
class buffer_reader : public mem_reader
{
private:
    py::buffer_info view;
public:
    explicit buffer_reader(stream &s, py::buffer_info &view)
        : mem_reader(s, reinterpret_cast<const std::uint8_t *>(view.ptr), view.itemsize * view.size),
        view(std::move(view))
    {
    }
};

class ring_stream_config_wrapper : public ring_stream_config
{
private:
    bool incomplete_keep_payload_ranges = false;

public:
    ring_stream_config_wrapper() = default;

    ring_stream_config_wrapper(const ring_stream_config &base) :
        ring_stream_config(base)
    {
    }

    ring_stream_config_wrapper &set_incomplete_keep_payload_ranges(bool keep)
    {
        incomplete_keep_payload_ranges = keep;
        return *this;
    }

    bool get_incomplete_keep_payload_ranges() const
    {
        return incomplete_keep_payload_ranges;
    }
};

#if SPEAD2_USE_IBV
/* Managing the endpoints and interface address requires some sleight of
 * hand. We store a separate copy in the wrapper in a Python-centric format.
 * When constructing the reader, we make a copy with the C++ view.
 */
class udp_ibv_config_wrapper : public udp_ibv_config
{
public:
    std::vector<std::pair<std::string, std::uint16_t>> py_endpoints;
    std::string py_interface_address;
};
#endif // SPEAD2_USE_IBV

/**
 * Stream that handles the magic necessary to reflect heaps into
 * Python space and capture the reference to it.
 *
 * The GIL needs to be handled carefully. Any operation run by the thread pool
 * might need to take the GIL to do logging. Thus, any operation that blocks
 * on completion of code scheduled through the thread pool must drop the GIL
 * first.
 */
class ring_stream_wrapper : public ring_stream<ringbuffer<live_heap, semaphore_gil<semaphore_fd>, semaphore> >
{
private:
    bool incomplete_keep_payload_ranges;
    exit_stopper stopper{[this] { stop(); }};

    boost::asio::ip::address make_address(const std::string &hostname)
    {
        return make_address_no_release(get_io_service(), hostname,
                                       boost::asio::ip::udp::resolver::query::passive);
    }

    template<typename Protocol>
    typename Protocol::endpoint make_endpoint(const std::string &hostname, std::uint16_t port)
    {
        return typename Protocol::endpoint(make_address(hostname), port);
    }

    py::object to_object(live_heap &&h)
    {
        if (h.is_contiguous())
            return py::cast(heap(std::move(h)), py::return_value_policy::move);
        else
            return py::cast(incomplete_heap(std::move(h), false, incomplete_keep_payload_ranges),
                            py::return_value_policy::move);
    }

public:
    ring_stream_wrapper(
        io_service_ref io_service,
        const stream_config &config = stream_config(),
        const ring_stream_config_wrapper &ring_config = ring_stream_config_wrapper())
        : ring_stream<ringbuffer<live_heap, semaphore_gil<semaphore_fd>, semaphore>>(
            std::move(io_service), config, ring_config),
        incomplete_keep_payload_ranges(ring_config.get_incomplete_keep_payload_ranges())
    {}

    py::object next()
    {
        try
        {
            return get();
        }
        catch (ringbuffer_stopped &e)
        {
            throw py::stop_iteration();
        }
    }

    py::object get()
    {
        return to_object(ring_stream::pop_live());
    }

    py::object get_nowait()
    {
        return to_object(try_pop_live());
    }

    int get_fd() const
    {
        return get_ringbuffer().get_data_sem().get_fd();
    }

    ring_stream_config_wrapper get_ring_config() const
    {
        ring_stream_config_wrapper ring_config(
            ring_stream<ringbuffer<live_heap, semaphore_gil<semaphore_fd>, semaphore> >::get_ring_config());
        ring_config.set_incomplete_keep_payload_ranges(incomplete_keep_payload_ranges);
        return ring_config;
    }

    void add_buffer_reader(py::buffer buffer)
    {
        py::buffer_info info = request_buffer_info(buffer, PyBUF_C_CONTIGUOUS);
        py::gil_scoped_release gil;
        emplace_reader<buffer_reader>(std::ref(info));
    }

    void add_udp_reader(
        std::uint16_t port,
        std::size_t max_size,
        std::size_t buffer_size,
        const std::string &bind_hostname)
    {
        py::gil_scoped_release gil;
        auto endpoint = make_endpoint<boost::asio::ip::udp>(bind_hostname, port);
        emplace_reader<udp_reader>(endpoint, max_size, buffer_size);
    }

    void add_udp_reader_socket(
        const socket_wrapper<boost::asio::ip::udp::socket> &socket,
        std::size_t max_size = udp_reader::default_max_size)
    {
        auto asio_socket = socket.copy(get_io_service());
        py::gil_scoped_release gil;
        emplace_reader<udp_reader>(std::move(asio_socket), max_size);
    }

    void add_udp_reader_bind_v4(
        const std::string &address,
        std::uint16_t port,
        std::size_t max_size,
        std::size_t buffer_size,
        const std::string &interface_address)
    {
        py::gil_scoped_release gil;
        auto endpoint = make_endpoint<boost::asio::ip::udp>(address, port);
        emplace_reader<udp_reader>(endpoint, max_size, buffer_size, make_address(interface_address));
    }

    void add_udp_reader_bind_v6(
        const std::string &address,
        std::uint16_t port,
        std::size_t max_size,
        std::size_t buffer_size,
        unsigned int interface_index)
    {
        py::gil_scoped_release gil;
        auto endpoint = make_endpoint<boost::asio::ip::udp>(address, port);
        emplace_reader<udp_reader>(endpoint, max_size, buffer_size, interface_index);
    }

    void add_tcp_reader(
        std::uint16_t port,
        std::size_t max_size,
        std::size_t buffer_size,
        const std::string &bind_hostname)
    {
        py::gil_scoped_release gil;
        auto endpoint = make_endpoint<boost::asio::ip::tcp>(bind_hostname, port);
        emplace_reader<tcp_reader>(endpoint, max_size, buffer_size);
    }

    void add_tcp_reader_socket(
        const socket_wrapper<boost::asio::ip::tcp::acceptor> &acceptor,
        std::size_t max_size)
    {
        auto asio_socket = acceptor.copy(get_io_service());
        py::gil_scoped_release gil;
        emplace_reader<tcp_reader>(std::move(asio_socket), max_size);
    }

#if SPEAD2_USE_IBV
    void add_udp_ibv_reader_single(
        const std::string &address,
        std::uint16_t port,
        const std::string &interface_address,
        std::size_t max_size,
        std::size_t buffer_size,
        int comp_vector,
        int max_poll)
    {
        deprecation_warning("Use a UdpIbvConfig instead");
        py::gil_scoped_release gil;
        auto endpoint = make_endpoint<boost::asio::ip::udp>(address, port);
        emplace_reader<udp_ibv_reader>(
            udp_ibv_config()
                .add_endpoint(endpoint)
                .set_interface_address(make_address(interface_address))
                .set_max_size(max_size)
                .set_buffer_size(buffer_size)
                .set_comp_vector(comp_vector)
                .set_max_poll(max_poll));
    }

    void add_udp_ibv_reader_multi(
        const py::sequence &endpoints,
        const std::string &interface_address,
        std::size_t max_size,
        std::size_t buffer_size,
        int comp_vector,
        int max_poll)
    {
        deprecation_warning("Use a UdpIbvConfig instead");
        // TODO: could this conversion be done by a custom caster?
        udp_ibv_config config;
        for (size_t i = 0; i < len(endpoints); i++)
        {
            py::sequence endpoint = endpoints[i].cast<py::sequence>();
            std::string address = endpoint[0].cast<std::string>();
            std::uint16_t port = endpoint[1].cast<std::uint16_t>();
            config.add_endpoint(make_endpoint<boost::asio::ip::udp>(address, port));
        }
        py::gil_scoped_release gil;
        config.set_interface_address(make_address(interface_address));
        config.set_max_size(max_size);
        config.set_buffer_size(buffer_size);
        config.set_comp_vector(comp_vector);
        config.set_max_poll(max_poll);
        emplace_reader<udp_ibv_reader>(config);
    }

    void add_udp_ibv_reader_new(const udp_ibv_config_wrapper &config_wrapper)
    {
        py::gil_scoped_release gil;
        udp_ibv_config config = config_wrapper;
        for (const auto &endpoint : config_wrapper.py_endpoints)
            config.add_endpoint(make_endpoint<boost::asio::ip::udp>(
                endpoint.first, endpoint.second));
        config.set_interface_address(
            make_address(config_wrapper.py_interface_address));
        emplace_reader<udp_ibv_reader>(config);
    }
#endif  // SPEAD2_USE_IBV

#if SPEAD2_USE_PCAP
    void add_udp_pcap_file_reader(const std::string &filename)
    {
        py::gil_scoped_release gil;
        emplace_reader<udp_pcap_file_reader>(filename);
    }
#endif

    void add_inproc_reader(std::shared_ptr<inproc_queue> queue)
    {
        py::gil_scoped_release gil;
        emplace_reader<inproc_reader>(queue);
    }

    void stop()
    {
        stopper.reset();
        py::gil_scoped_release gil;
        ring_stream::stop();
    }

    ~ring_stream_wrapper()
    {
        stop();
    }
};

/// Register the receiver module with Boost.Python
py::module register_module(py::module &parent)
{
    using namespace pybind11::literals;

    // Create the module
    py::module m = parent.def_submodule("recv");

    py::class_<heap_base>(m, "HeapBase")
        .def_property_readonly("cnt", SPEAD2_PTMF(heap_base, get_cnt))
        .def_property_readonly("flavour", SPEAD2_PTMF(heap_base, get_flavour))
        .def("get_items", [](py::object &self) -> py::list
        {
            const heap_base &h = self.cast<const heap_base &>();
            std::vector<item> base = h.get_items();
            py::list out;
            for (const item &it : base)
            {
                // Filter out descriptors here. The base class can't do so, because
                // the descriptors are retrieved from the items.
                if (it.id != DESCRIPTOR_ID)
                    out.append(item_wrapper(it, self));
            }
            return out;
        })
        .def("is_start_of_stream", SPEAD2_PTMF(heap_base, is_start_of_stream))
        .def("is_end_of_stream", SPEAD2_PTMF(heap_base, is_end_of_stream));
    py::class_<heap, heap_base>(m, "Heap")
        .def("get_descriptors", SPEAD2_PTMF(heap, get_descriptors));
    py::class_<incomplete_heap, heap_base>(m, "IncompleteHeap")
        .def_property_readonly("heap_length", SPEAD2_PTMF(incomplete_heap, get_heap_length))
        .def_property_readonly("received_length", SPEAD2_PTMF(incomplete_heap, get_received_length))
        .def_property_readonly("payload_ranges", SPEAD2_PTMF(incomplete_heap, get_payload_ranges));
    py::class_<item_wrapper>(m, "RawItem", py::buffer_protocol())
        .def_readonly("id", &item_wrapper::id)
        .def_readonly("is_immediate", &item_wrapper::is_immediate)
        .def_readonly("immediate_value", &item_wrapper::immediate_value)
        .def_buffer([](item_wrapper &item) { return item.get_value(); });
    py::class_<stream_stats>(m, "StreamStats")
        .def_readwrite("heaps", &stream_stats::heaps)
        .def_readwrite("incomplete_heaps_evicted", &stream_stats::incomplete_heaps_evicted)
        .def_readwrite("incomplete_heaps_flushed", &stream_stats::incomplete_heaps_flushed)
        .def_readwrite("packets", &stream_stats::packets)
        .def_readwrite("batches", &stream_stats::batches)
        .def_readwrite("worker_blocked", &stream_stats::worker_blocked)
        .def_readwrite("max_batch", &stream_stats::max_batch)
        .def_readwrite("single_packet_heaps", &stream_stats::single_packet_heaps)
        .def_readwrite("search_dist", &stream_stats::search_dist)
        .def(py::self + py::self)
        .def(py::self += py::self);
    py::class_<stream_config>(m, "StreamConfig")
        .def(py::init(&data_class_constructor<stream_config>))
        .def_property("max_heaps",
                      SPEAD2_PTMF(stream_config, get_max_heaps),
                      SPEAD2_PTMF(stream_config, set_max_heaps))
        .def_property("bug_compat",
                      SPEAD2_PTMF(stream_config, get_bug_compat),
                      SPEAD2_PTMF(stream_config, set_bug_compat))
        .def_property("memcpy",
             [](const stream_config &self) {
                 stream_config cmp;
                 memcpy_function_id ids[] = {MEMCPY_STD, MEMCPY_NONTEMPORAL};
                 for (memcpy_function_id id : ids)
                 {
                     cmp.set_memcpy(id);
                     if (equal_functions(self.get_memcpy(), cmp.get_memcpy()))
                         return int(id);
                 }
                 throw std::invalid_argument("memcpy function is not one of the standard ones");
             },
             [](stream_config &self, int id) { self.set_memcpy(memcpy_function_id(id)); })
        .def_property("memory_allocator",
                      SPEAD2_PTMF(stream_config, get_memory_allocator),
                      SPEAD2_PTMF_VOID(stream_config, set_memory_allocator))
        .def_property("stop_on_stop_item",
                      SPEAD2_PTMF(stream_config, get_stop_on_stop_item),
                      SPEAD2_PTMF_VOID(stream_config, set_stop_on_stop_item))
        .def_property("allow_unsized_heaps",
                      SPEAD2_PTMF(stream_config, get_allow_unsized_heaps),
                      SPEAD2_PTMF_VOID(stream_config, set_allow_unsized_heaps))
        .def_property("allow_out_of_order",
                      SPEAD2_PTMF(stream_config, get_allow_out_of_order),
                      SPEAD2_PTMF_VOID(stream_config, set_allow_out_of_order))
        .def_readonly_static("DEFAULT_MAX_HEAPS", &stream_config::default_max_heaps);
    py::class_<ring_stream_config_wrapper>(m, "RingStreamConfig")
        .def(py::init(&data_class_constructor<ring_stream_config_wrapper>))
        .def_property("heaps",
                      SPEAD2_PTMF(ring_stream_config_wrapper, get_heaps),
                      SPEAD2_PTMF_VOID(ring_stream_config_wrapper, set_heaps))
        .def_property("contiguous_only",
                      SPEAD2_PTMF(ring_stream_config_wrapper, get_contiguous_only),
                      SPEAD2_PTMF_VOID(ring_stream_config_wrapper, set_contiguous_only))
        .def_property("incomplete_keep_payload_ranges",
                      SPEAD2_PTMF(ring_stream_config_wrapper, get_incomplete_keep_payload_ranges),
                      SPEAD2_PTMF_VOID(ring_stream_config_wrapper, set_incomplete_keep_payload_ranges))
        .def_readonly_static("DEFAULT_HEAPS", &ring_stream_config_wrapper::default_heaps);
#if SPEAD2_USE_IBV
    py::class_<udp_ibv_config_wrapper>(m, "UdpIbvConfig")
        .def(py::init(&data_class_constructor<udp_ibv_config_wrapper>))
        .def_readwrite("endpoints", &udp_ibv_config_wrapper::py_endpoints)
        .def_readwrite("interface_address", &udp_ibv_config_wrapper::py_interface_address)
        .def_property("buffer_size",
                      SPEAD2_PTMF(udp_ibv_config_wrapper, get_buffer_size),
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_buffer_size))
        .def_property("max_size",
                      SPEAD2_PTMF(udp_ibv_config_wrapper, get_max_size),
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_max_size))
        .def_property("comp_vector",
                      SPEAD2_PTMF(udp_ibv_config_wrapper, get_comp_vector),
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_comp_vector))
        .def_property("max_poll",
                      SPEAD2_PTMF(udp_ibv_config_wrapper, get_max_poll),
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_max_poll))
        .def_readonly_static("DEFAULT_BUFFER_SIZE", &udp_ibv_config_wrapper::default_buffer_size)
        .def_readonly_static("DEFAULT_MAX_SIZE", &udp_ibv_config_wrapper::default_max_size)
        .def_readonly_static("DEFAULT_MAX_POLL", &udp_ibv_config_wrapper::default_max_poll);
#endif // SPEAD2_USE_IBV
    py::class_<ring_stream_wrapper> stream_class(m, "Stream");
    stream_class
        .def(py::init<std::shared_ptr<thread_pool_wrapper>,
                      const stream_config &,
                      const ring_stream_config_wrapper &>(),
             "thread_pool"_a, "config"_a = stream_config(),
             "ring_config"_a = ring_stream_config_wrapper())
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", SPEAD2_PTMF(ring_stream_wrapper, next))
        .def("get", SPEAD2_PTMF(ring_stream_wrapper, get))
        .def("get_nowait", SPEAD2_PTMF(ring_stream_wrapper, get_nowait))
        .def("add_buffer_reader", SPEAD2_PTMF(ring_stream_wrapper, add_buffer_reader), "buffer"_a)
        .def("add_udp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_reader),
              "port"_a,
              "max_size"_a = udp_reader::default_max_size,
              "buffer_size"_a = udp_reader::default_buffer_size,
              "bind_hostname"_a = std::string())
        .def("add_udp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_reader_socket),
              "socket"_a,
              "max_size"_a = udp_reader::default_max_size)
        .def("add_udp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_reader_bind_v4),
              "multicast_group"_a,
              "port"_a,
              "max_size"_a = udp_reader::default_max_size,
              "buffer_size"_a = udp_reader::default_buffer_size,
              "interface_address"_a = "0.0.0.0")
        .def("add_udp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_reader_bind_v6),
              "multicast_group"_a,
              "port"_a,
              "max_size"_a = udp_reader::default_max_size,
              "buffer_size"_a = udp_reader::default_buffer_size,
              "interface_index"_a = (unsigned int) 0)
        .def("add_tcp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_tcp_reader),
             "port"_a,
             "max_size"_a = tcp_reader::default_max_size,
             "buffer_size"_a = tcp_reader::default_buffer_size,
             "bind_hostname"_a = std::string())
        .def("add_tcp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_tcp_reader_socket),
             "acceptor"_a,
             "max_size"_a = tcp_reader::default_max_size)
#if SPEAD2_USE_IBV
        .def("add_udp_ibv_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_ibv_reader_single),
              "multicast_group"_a,
              "port"_a,
              "interface_address"_a,
              "max_size"_a = udp_ibv_config::default_max_size,
              "buffer_size"_a = udp_ibv_config::default_buffer_size,
              "comp_vector"_a = 0,
              "max_poll"_a = udp_ibv_config::default_max_poll)
        .def("add_udp_ibv_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_ibv_reader_multi),
              "endpoints"_a,
              "interface_address"_a,
              "max_size"_a = udp_ibv_config::default_max_size,
              "buffer_size"_a = udp_ibv_config::default_buffer_size,
              "comp_vector"_a = 0,
              "max_poll"_a = udp_ibv_config::default_max_poll)
        .def("add_udp_ibv_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_ibv_reader_new),
             "config"_a)
#endif
#if SPEAD2_USE_PCAP
        .def("add_udp_pcap_file_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_pcap_file_reader),
             "filename"_a)
#endif
        .def("add_inproc_reader", SPEAD2_PTMF(ring_stream_wrapper, add_inproc_reader),
             "queue"_a)
        .def("stop", SPEAD2_PTMF(ring_stream_wrapper, stop))
        .def_property_readonly("fd", SPEAD2_PTMF(ring_stream_wrapper, get_fd))
        // SPEAD2_PTMF doesn't work for get_stats because it's defined in stream_base, which is a protected ancestor
        .def_property_readonly("stats", [](const ring_stream_wrapper &stream) { return stream.get_stats(); })
        .def_property_readonly("ringbuffer", SPEAD2_PTMF(ring_stream_wrapper, get_ringbuffer))
        .def_property_readonly("config",
                               [](const ring_stream_wrapper &self) { return self.get_config(); })
        .def_property_readonly("ring_config", SPEAD2_PTMF(ring_stream_wrapper, get_ring_config))
#if SPEAD2_USE_IBV
        .def_property_readonly_static("DEFAULT_UDP_IBV_MAX_SIZE",
            [](py::object) {
                deprecation_warning("Use spead2.recv.UdpIbvConfig.DEFAULT_MAX_SIZE");
                return udp_ibv_config::default_max_size;
            })
        .def_property_readonly_static("DEFAULT_UDP_IBV_BUFFER_SIZE",
            [](py::object) {
                deprecation_warning("Use spead2.recv.UdpIbvConfig.DEFAULT_BUFFER_SIZE");
                return udp_ibv_config::default_buffer_size;
            })
        .def_property_readonly_static("DEFAULT_UDP_IBV_MAX_POLL",
            [](py::object) {
                deprecation_warning("Use spead2.recv.UdpIbvConfig.DEFAULT_MAX_POLL");
                return udp_ibv_config::default_max_poll;
            })
#endif
        .def_readonly_static("DEFAULT_UDP_MAX_SIZE", &udp_reader::default_max_size)
        .def_readonly_static("DEFAULT_UDP_BUFFER_SIZE", &udp_reader::default_buffer_size)
        .def_readonly_static("DEFAULT_TCP_MAX_SIZE", &tcp_reader::default_max_size)
        .def_readonly_static("DEFAULT_TCP_BUFFER_SIZE", &tcp_reader::default_buffer_size);
    using Ringbuffer = ringbuffer<live_heap, semaphore_gil<semaphore_fd>, semaphore>;
    py::class_<Ringbuffer>(stream_class, "Ringbuffer")
        .def("size", SPEAD2_PTMF(Ringbuffer, size))
        .def("capacity", SPEAD2_PTMF(Ringbuffer, capacity));

    return m;
}

} // namespace recv
} // namespace spead2
