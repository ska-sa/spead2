/* Copyright 2015, 2017 SKA South Africa
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
        bug_compat_mask bug_compat = 0,
        std::size_t max_heaps = default_max_heaps,
        std::size_t ring_heaps = default_ring_heaps,
        bool contiguous_only = true,
        bool incomplete_keep_payload_ranges = false)
        : ring_stream<ringbuffer<live_heap, semaphore_gil<semaphore_fd>, semaphore>>(
            std::move(io_service), bug_compat, max_heaps, ring_heaps, contiguous_only),
        incomplete_keep_payload_ranges(incomplete_keep_payload_ranges)
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

    void set_memory_pool(std::shared_ptr<memory_pool> pool)
    {
        py::gil_scoped_release gil;
        ring_stream::set_memory_allocator(std::move(pool));
    }

    void set_memory_allocator(std::shared_ptr<memory_allocator> allocator)
    {
        py::gil_scoped_release gil;
        ring_stream::set_memory_allocator(std::move(allocator));
    }

    void set_memcpy(int id)
    {
        ring_stream::set_memcpy(memcpy_function_id(id));
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
        const std::string &bind_hostname,
        const boost::optional<socket_wrapper<boost::asio::ip::udp::socket>> &socket)
    {
        if (!socket)
        {
            py::gil_scoped_release gil;
            auto endpoint = make_endpoint<boost::asio::ip::udp>(bind_hostname, port);
            emplace_reader<udp_reader>(endpoint, max_size, buffer_size);
        }
        else
        {
            deprecation_warning("passing unbound socket plus port is deprecated");
            auto asio_socket = socket->copy(get_io_service());
            py::gil_scoped_release gil;
            auto endpoint = make_endpoint<boost::asio::ip::udp>(bind_hostname, port);
            emplace_reader<udp_reader>(std::move(asio_socket), endpoint, max_size, buffer_size);
        }
    }

    void add_udp_reader_socket(
        const socket_wrapper<boost::asio::ip::udp::socket> &socket,
        std::size_t max_size = udp_reader::default_max_size)
    {
        auto asio_socket = socket.copy(get_io_service());
        py::gil_scoped_release gil;
        emplace_reader<udp_reader>(std::move(asio_socket), max_size);
    }

    void add_udp_reader_multicast_v4(
        const std::string &multicast_group,
        std::uint16_t port,
        std::size_t max_size,
        std::size_t buffer_size,
        const std::string &interface_address)
    {
        py::gil_scoped_release gil;
        auto endpoint = make_endpoint<boost::asio::ip::udp>(multicast_group, port);
        emplace_reader<udp_reader>(endpoint, max_size, buffer_size, make_address(interface_address));
    }

    void add_udp_reader_multicast_v6(
        const std::string &multicast_group,
        std::uint16_t port,
        std::size_t max_size,
        std::size_t buffer_size,
        unsigned int interface_index)
    {
        py::gil_scoped_release gil;
        auto endpoint = make_endpoint<boost::asio::ip::udp>(multicast_group, port);
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
        const std::string &multicast_group,
        std::uint16_t port,
        const std::string &interface_address,
        std::size_t max_size,
        std::size_t buffer_size,
        int comp_vector,
        int max_poll)
    {
        py::gil_scoped_release gil;
        auto endpoint = make_endpoint<boost::asio::ip::udp>(multicast_group, port);
        emplace_reader<udp_ibv_reader>(endpoint, make_address(interface_address),
                                       max_size, buffer_size, comp_vector, max_poll);
    }

    void add_udp_ibv_reader_multi(
        const py::sequence &endpoints,
        const std::string &interface_address,
        std::size_t max_size,
        std::size_t buffer_size,
        int comp_vector,
        int max_poll)
    {
        // TODO: could this conversion be done by a custom caster?
        std::vector<boost::asio::ip::udp::endpoint> endpoints2;
        for (size_t i = 0; i < len(endpoints); i++)
        {
            py::sequence endpoint = endpoints[i].cast<py::sequence>();
            std::string multicast_group = endpoint[0].cast<std::string>();
            std::uint16_t port = endpoint[1].cast<std::uint16_t>();
            endpoints2.push_back(make_endpoint<boost::asio::ip::udp>(multicast_group, port));
        }
        py::gil_scoped_release gil;
        emplace_reader<udp_ibv_reader>(endpoints2, make_address(interface_address),
                                       max_size, buffer_size, comp_vector, max_poll);
    }
#endif

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
    using namespace spead2::recv;

    // Create the module, and set it as the current boost::python scope so that
    // classes we define are added to this module rather than the root.
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
    py::class_<ring_stream_wrapper> stream_class(m, "Stream");
    stream_class
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, bug_compat_mask,
                      std::size_t, std::size_t, bool, bool>(),
             "thread_pool"_a, "bug_compat"_a = 0,
             "max_heaps"_a = ring_stream_wrapper::default_max_heaps,
             "ring_heaps"_a = ring_stream_wrapper::default_ring_heaps,
             "contiguous_only"_a = true,
             "incomplete_keep_payload_ranges"_a = false)
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", SPEAD2_PTMF(ring_stream_wrapper, next))
        .def("get", SPEAD2_PTMF(ring_stream_wrapper, get))
        .def("get_nowait", SPEAD2_PTMF(ring_stream_wrapper, get_nowait))
        .def("set_memory_allocator", SPEAD2_PTMF(ring_stream_wrapper, set_memory_allocator),
             "allocator"_a)
        .def("set_memory_pool", SPEAD2_PTMF(ring_stream_wrapper, set_memory_pool),
             "pool"_a)
        .def("set_memcpy", SPEAD2_PTMF(ring_stream_wrapper, set_memcpy), "id"_a)
        .def_property("stop_on_stop_item",
                      /* SPEAD2_PTMF doesn't work here because the functions
                       * are defined in stream_base, which is a private base
                       * class, and only made accessible via "using".
                       */
                      [](const ring_stream_wrapper &self) {
                          return self.get_stop_on_stop_item();
                      },
                      [](ring_stream_wrapper &self, bool stop) {
                          self.set_stop_on_stop_item(stop);
                      })
        .def_property("allow_unsized_heaps",
                      [](const ring_stream_wrapper &self) {
                          return self.get_allow_unsized_heaps();
                      },
                      [](ring_stream_wrapper &self, bool allow) {
                          self.set_allow_unsized_heaps(allow);
                      })
        .def("add_buffer_reader", SPEAD2_PTMF(ring_stream_wrapper, add_buffer_reader), "buffer"_a)
        .def("add_udp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_reader),
              "port"_a,
              "max_size"_a = udp_reader::default_max_size,
              "buffer_size"_a = udp_reader::default_buffer_size,
              "bind_hostname"_a = std::string(),
              "socket"_a = py::none())
        .def("add_udp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_reader_socket),
              "socket"_a,
              "max_size"_a = udp_reader::default_max_size)
        .def("add_udp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_reader_multicast_v4),
              "multicast_group"_a,
              "port"_a,
              "max_size"_a = udp_reader::default_max_size,
              "buffer_size"_a = udp_reader::default_buffer_size,
              "interface_address"_a = "0.0.0.0")
        .def("add_udp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_reader_multicast_v6),
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
              "max_size"_a = udp_ibv_reader::default_max_size,
              "buffer_size"_a = udp_ibv_reader::default_buffer_size,
              "comp_vector"_a = 0,
              "max_poll"_a = udp_ibv_reader::default_max_poll)
        .def("add_udp_ibv_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_ibv_reader_multi),
              "endpoints"_a,
              "interface_address"_a,
              "max_size"_a = udp_ibv_reader::default_max_size,
              "buffer_size"_a = udp_ibv_reader::default_buffer_size,
              "comp_vector"_a = 0,
              "max_poll"_a = udp_ibv_reader::default_max_poll)
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
#if SPEAD2_USE_IBV
        .def_readonly_static("DEFAULT_UDP_IBV_MAX_SIZE", &udp_ibv_reader::default_max_size)
        .def_readonly_static("DEFAULT_UDP_IBV_BUFFER_SIZE", &udp_ibv_reader::default_buffer_size)
        .def_readonly_static("DEFAULT_UDP_IBV_MAX_POLL", &udp_ibv_reader::default_max_poll)
#endif
        .def_readonly_static("DEFAULT_MAX_HEAPS", &ring_stream_wrapper::default_max_heaps)
        .def_readonly_static("DEFAULT_RING_HEAPS", &ring_stream_wrapper::default_ring_heaps)
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
