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
#include <stdexcept>
#include <cstdint>
#include <unistd.h>
#include <spead2/recv_udp.h>
#include <spead2/recv_udp_ibv.h>
#include <spead2/recv_mem.h>
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
    boost::asio::ip::address make_address(const std::string &hostname)
    {
        if (hostname.empty())
            return boost::asio::ip::address_v4::any();
        else
        {
            using boost::asio::ip::udp;
            udp::resolver resolver(get_strand().get_io_service());
            udp::resolver::query query(hostname, "", udp::resolver::query::passive);
            return resolver.resolve(query)->endpoint().address();
        }
    }

    boost::asio::ip::udp::endpoint make_endpoint(const std::string &hostname, std::uint16_t port)
    {
        return boost::asio::ip::udp::endpoint(make_address(hostname), port);
    }

public:
    using ring_stream::ring_stream;

    heap next()
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

    heap get()
    {
        return ring_stream::pop();
    }

    heap get_nowait()
    {
        return try_pop();
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
        std::size_t max_size = udp_reader::default_max_size,
        std::size_t buffer_size = udp_reader::default_buffer_size,
        const std::string &bind_hostname = "",
        const py::object &socket = py::none())
    {
        int fd2 = -1;
        if (!socket.is_none())
        {
            int fd = socket.attr("fileno")().cast<int>();
            /* Python still owns this FD and will close it, so we have to duplicate
             * it for ourselves.
             */
            fd2 = ::dup(fd);
            if (fd2 == -1)
            {
                PyErr_SetFromErrno(PyExc_OSError);
                throw py::error_already_set();
            }
        }

        py::gil_scoped_release gil;
        auto endpoint = make_endpoint(bind_hostname, port);
        if (fd2 == -1)
        {
            emplace_reader<udp_reader>(endpoint, max_size, buffer_size);
        }
        else
        {
            boost::asio::ip::udp::socket asio_socket(
                get_strand().get_io_service(), endpoint.protocol(), fd2);
            emplace_reader<udp_reader>(std::move(asio_socket), endpoint, max_size, buffer_size);
        }
    }

    void add_udp_reader_multicast_v4(
        const std::string &multicast_group,
        std::uint16_t port,
        std::size_t max_size,
        std::size_t buffer_size,
        const std::string &interface_address)
    {
        py::gil_scoped_release gil;
        auto endpoint = make_endpoint(multicast_group, port);
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
        auto endpoint = make_endpoint(multicast_group, port);
        emplace_reader<udp_reader>(endpoint, max_size, buffer_size, interface_index);
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
        auto endpoint = make_endpoint(multicast_group, port);
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
            endpoints2.push_back(make_endpoint(multicast_group, port));
        }
        py::gil_scoped_release gil;
        emplace_reader<udp_ibv_reader>(endpoints2, make_address(interface_address),
                                       max_size, buffer_size, comp_vector, max_poll);
    }
#endif

    void stop()
    {
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

    spead2::class_<heap>(m, "Heap")
        .def_property_readonly("cnt", SPEAD2_PTMF(heap, get_cnt))
        .def_property_readonly("flavour", SPEAD2_PTMF(heap, get_flavour))
        .def("get_items", [](const heap &h) -> py::list
        {
            std::vector<item> base = h.get_items();
            py::list out;
            py::object self = py::cast(&h);
            for (const item &it : base)
            {
                // Filter out descriptors here. The base class can't do so, because
                // the descriptors are retrieved from the items.
                if (it.id != DESCRIPTOR_ID)
                    out.append(item_wrapper(it, self));
            }
            return out;
        })
        .def("get_descriptors", SPEAD2_PTMF(heap, get_descriptors))
        .def("is_start_of_stream", SPEAD2_PTMF(heap, is_start_of_stream));
    spead2::class_<item_wrapper>(m, "RawItem", py::buffer_protocol())
        .def_readonly("id", &item_wrapper::id)
        .def_readonly("is_immediate", &item_wrapper::is_immediate)
        .def_readonly("immediate_value", &item_wrapper::immediate_value)
        .def_buffer([](item_wrapper &item) { return item.get_value(); });
    spead2::class_<ring_stream_wrapper>(m, "Stream")
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, bug_compat_mask, std::size_t, std::size_t>(),
             "thread_pool"_a, "bug_compat"_a = 0,
             "max_heaps"_a = ring_stream_wrapper::default_max_heaps,
             "ring_heaps"_a = ring_stream_wrapper::default_ring_heaps)
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", SPEAD2_PTMF(ring_stream_wrapper, next))
        .def("get", SPEAD2_PTMF(ring_stream_wrapper, get))
        .def("get_nowait", SPEAD2_PTMF(ring_stream_wrapper, get_nowait))
        .def("set_memory_allocator", SPEAD2_PTMF(ring_stream_wrapper, set_memory_allocator),
             "allocator"_a)
        .def("set_memory_pool", SPEAD2_PTMF(ring_stream_wrapper, set_memory_pool),
             "pool"_a)
        .def("set_memcpy", SPEAD2_PTMF(ring_stream_wrapper, set_memcpy), "id"_a)
        .def("add_buffer_reader", SPEAD2_PTMF(ring_stream_wrapper, add_buffer_reader), "buffer"_a)
        .def("add_udp_reader", SPEAD2_PTMF(ring_stream_wrapper, add_udp_reader),
              "port"_a,
              "max_size"_a = udp_reader::default_max_size,
              "buffer_size"_a = udp_reader::default_buffer_size,
              "bind_hostname"_a = std::string(),
              "socket"_a = py::none())
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
        .def("stop", SPEAD2_PTMF(ring_stream_wrapper, stop))
        .def_property_readonly("fd", SPEAD2_PTMF(ring_stream_wrapper, get_fd))
#if SPEAD2_USE_IBV
        .def_readonly_static("DEFAULT_UDP_IBV_MAX_SIZE", &udp_ibv_reader::default_max_size)
        .def_readonly_static("DEFAULT_UDP_IBV_BUFFER_SIZE", &udp_ibv_reader::default_buffer_size)
        .def_readonly_static("DEFAULT_UDP_IBV_MAX_POLL", &udp_ibv_reader::default_max_poll)
#endif
        .def_readonly_static("DEFAULT_MAX_HEAPS", &ring_stream_wrapper::default_max_heaps)
        .def_readonly_static("DEFAULT_RING_HEAPS", &ring_stream_wrapper::default_ring_heaps)
        .def_readonly_static("DEFAULT_UDP_MAX_SIZE", &udp_reader::default_max_size)
        .def_readonly_static("DEFAULT_UDP_BUFFER_SIZE", &udp_reader::default_buffer_size);

    return m;
}

} // namespace recv
} // namespace spead2
