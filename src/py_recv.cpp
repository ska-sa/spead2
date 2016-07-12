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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL spead2_ARRAY_API
#define NO_IMPORT_ARRAY
#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include <numpy/arrayobject.h>
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

namespace py = boost::python;

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
    py::handle<> owning_heap;

public:
    item_wrapper() = default;
    item_wrapper(const item &it, PyObject *owning_heap)
        : item(it), owning_heap(py::borrowed(owning_heap)) {}

    /**
     * Obtain the raw value, as a Python object memoryview.  Since
     * boost::python doesn't support creating classes with the buffer protocol,
     * we need a proxy object to wrap in the memoryview, and it needs a way to
     * hold a reference on the owning heap. PyBuffer_* doesn't satisfy the
     * second requirement, but numpy arrays do. It's rather overkill, but the
     * package uses numpy anyway.
     */
    py::object get_value() const
    {
        npy_intp dims[1];
        // Construct the numpy array to wrap our data
        dims[0] = length;
        PyObject *array = PyArray_SimpleNewFromData(
            1, dims, NPY_UINT8,
            reinterpret_cast<void *>(ptr));
        if (array == NULL)
            py::throw_error_already_set();
        // Make the numpy array hold a ref to the owning heap
        if (PyArray_SetBaseObject(
                (PyArrayObject *) array,
                py::incref(owning_heap.get())) == -1)
        {
            py::decref(owning_heap.get());
            py::throw_error_already_set();
        }

        // Create a memoryview wrapper around the numpy array
        PyObject *view = PyMemoryView_FromObject(array);
        if (view == NULL)
        {
            py::decref(array); // discard the array
            py::throw_error_already_set();
        }
        py::object obj{py::handle<>(view)};
        py::decref(array); // the view holds its own ref
        return obj;
    }
};

/**
 * Wrapper for heaps. This is used as a held type, so that it gets a back
 * reference to the Python object (needed so that item_wrapper can hold a
 * reference to the heap).
 *
 * The constructor steals data from the provided heap, but Boost.Python
 * doesn't currently understand C++11 rvalue semantics. Thus, it is vital
 * that any wrapped function returning a heap returns a temporary that can be
 * safely clobbered.
 */
class heap_wrapper : public heap
{
private:
    PyObject *self;

public:
    heap_wrapper(PyObject *self, const heap &h)
        : heap(std::move(const_cast<heap &>(h))), self(self) {}

    /// Wrap @ref heap::get_items, and convert vector to a Python list
    py::list get_items() const
    {
        std::vector<item> base = heap::get_items();
        py::list out;
        for (const item &it : base)
        {
            // Filter out descriptors here. The base class can't do so, because
            // the descriptors are retrieved from the items.
            if (it.id != DESCRIPTOR_ID)
                out.append(item_wrapper(it, self));
        }
        return out;
    }

    /// Wrap @ref heap::get_descriptors, and convert vector to a Python list
    py::list get_descriptors() const
    {
        std::vector<descriptor> descriptors = heap::get_descriptors();
        py::list out;
        for (const descriptor& d : descriptors)
            out.append(d);
        return out;
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
    buffer_view view;
public:
    explicit buffer_reader(stream &s, buffer_view &view)
        : mem_reader(s, reinterpret_cast<const std::uint8_t *>(view.view.buf), view.view.len),
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
class ring_stream_wrapper : public thread_pool_handle_wrapper,
                            public memory_allocator_handle_wrapper,
                            public ring_stream<ringbuffer<live_heap, semaphore_gil<semaphore_fd>, semaphore> >
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
            throw stop_iteration();
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

    void set_memory_pool(std::shared_ptr<memory_pool_wrapper> pool)
    {
        release_gil gil;
        ring_stream::set_memory_allocator(std::move(pool));
    }

    void set_memory_allocator(std::shared_ptr<memory_allocator> allocator)
    {
        release_gil gil;
        ring_stream::set_memory_allocator(std::move(allocator));
    }

    void set_memcpy(int id)
    {
        ring_stream::set_memcpy(memcpy_function_id(id));
    }

    void add_buffer_reader(py::object buffer)
    {
        buffer_view view(buffer);
        release_gil gil;
        emplace_reader<buffer_reader>(std::ref(view));
    }

    void add_udp_reader(
        std::uint16_t port,
        std::size_t max_size = udp_reader::default_max_size,
        std::size_t buffer_size = udp_reader::default_buffer_size,
        const std::string &bind_hostname = "",
        const py::object &socket = py::object())
    {
        int fd2 = -1;
        if (!socket.is_none())
        {
            int fd = py::extract<int>(socket.attr("fileno")());
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

        release_gil gil;
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
        release_gil gil;
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
        release_gil gil;
        auto endpoint = make_endpoint(multicast_group, port);
        emplace_reader<udp_reader>(endpoint, max_size, buffer_size, interface_index);
    }

#if SPEAD2_USE_IBV
    void add_udp_ibv_reader(
        const std::string &multicast_group,
        std::uint16_t port,
        const std::string &interface_address,
        std::size_t max_size,
        std::size_t buffer_size,
        int comp_vector,
        int max_poll)
    {
        release_gil gil;
        auto endpoint = make_endpoint(multicast_group, port);
        emplace_reader<udp_ibv_reader>(endpoint, make_address(interface_address),
                                       max_size, buffer_size, comp_vector, max_poll);
    }
#endif

    void stop()
    {
        release_gil gil;
        ring_stream::stop();
    }

    ~ring_stream_wrapper()
    {
        stop();
    }
};

/// Register the receiver module with Boost.Python
void register_module()
{
    using namespace boost::python;
    using namespace spead2::recv;

    // Create the module, and set it as the current boost::python scope so that
    // classes we define are added to this module rather than the root.
    py::object module(py::handle<>(py::borrowed(PyImport_AddModule("spead2._recv"))));
    py::scope scope = module;

    class_<heap, heap_wrapper>("Heap", no_init)
        .add_property("cnt", &heap_wrapper::get_cnt)
        .add_property("flavour",
            make_function(&heap_wrapper::get_flavour, return_value_policy<copy_const_reference>()))
        .def("get_items", &heap_wrapper::get_items)
        .def("get_descriptors", &heap_wrapper::get_descriptors)
        .def("is_start_of_stream", &heap_wrapper::is_start_of_stream);
    class_<item_wrapper>("RawItem", no_init)
        .def_readonly("id", &item_wrapper::id)
        .def_readonly("is_immediate", &item_wrapper::is_immediate)
        .def_readonly("immediate_value", &item_wrapper::immediate_value)
        .add_property("value", &item_wrapper::get_value);
    class_<ring_stream_wrapper, boost::noncopyable>("Stream",
            init<thread_pool_wrapper &, bug_compat_mask, std::size_t, std::size_t>(
                (arg("thread_pool"), arg("bug_compat") = 0,
                 arg("max_heaps") = ring_stream_wrapper::default_max_heaps,
                 arg("ring_heaps") = ring_stream_wrapper::default_ring_heaps))[
                store_handle_postcall<ring_stream_wrapper, thread_pool_handle_wrapper, &thread_pool_handle_wrapper::thread_pool_handle, 1, 2>()])
        .def("__iter__", objects::identity_function())
        .def(
#if PY_MAJOR_VERSION >= 3
              // Python 3 uses __next__ for the iterator protocol
              "__next__"
#else
              "next"
#endif
        , &ring_stream_wrapper::next)
        .def("get", &ring_stream_wrapper::get)
        .def("get_nowait", &ring_stream_wrapper::get_nowait)
        .def("set_memory_allocator", &ring_stream_wrapper::set_memory_allocator,
             (arg("allocator")),
             store_handle_postcall<ring_stream_wrapper, memory_allocator_handle_wrapper, &memory_allocator_handle_wrapper::memory_allocator_handle, 1, 2>())
        .def("set_memory_pool", &ring_stream_wrapper::set_memory_pool,
             (arg("pool")),
             store_handle_postcall<ring_stream_wrapper, memory_allocator_handle_wrapper, &memory_allocator_handle_wrapper::memory_allocator_handle, 1, 2>())
        .def("set_memcpy", &ring_stream_wrapper::set_memcpy,
             arg("id"))
        .def("add_buffer_reader", &ring_stream_wrapper::add_buffer_reader,
             arg("buffer"))
        .def("add_udp_reader", &ring_stream_wrapper::add_udp_reader,
             (arg("port"),
              arg("max_size") = udp_reader::default_max_size,
              arg("buffer_size") = udp_reader::default_buffer_size,
              arg("bind_hostname") = std::string(),
              arg("socket") = py::object()))
        .def("add_udp_reader", &ring_stream_wrapper::add_udp_reader_multicast_v4,
             (
              arg("multicast_group"),
              arg("port"),
              arg("max_size") = udp_reader::default_max_size,
              arg("buffer_size") = udp_reader::default_buffer_size,
              arg("interface_address") = "0.0.0.0"))
        .def("add_udp_reader", &ring_stream_wrapper::add_udp_reader_multicast_v6,
             (
              arg("multicast_group"),
              arg("port"),
              arg("max_size") = udp_reader::default_max_size,
              arg("buffer_size") = udp_reader::default_buffer_size,
              arg("interface_index") = (unsigned int) 0))
#if SPEAD2_USE_IBV
        .def("add_udp_ibv_reader", &ring_stream_wrapper::add_udp_ibv_reader,
             (
              arg("multicast_group"),
              arg("port"),
              arg("interface_address"),
              arg("max_size") = udp_ibv_reader::default_max_size,
              arg("buffer_size") = udp_ibv_reader::default_buffer_size,
              arg("comp_vector") = 0,
              arg("max_poll") = udp_ibv_reader::default_max_poll))
#endif
        .def("stop", &ring_stream_wrapper::stop)
        .add_property("fd", &ring_stream_wrapper::get_fd)
#if SPEAD2_USE_IBV
        .def_readonly("DEFAULT_UDP_IBV_MAX_SIZE", udp_ibv_reader::default_max_size)
        .def_readonly("DEFAULT_UDP_IBV_BUFFER_SIZE", udp_ibv_reader::default_buffer_size)
        .def_readonly("DEFAULT_UDP_IBV_MAX_POLL", udp_ibv_reader::default_max_poll)
#endif
        .def_readonly("DEFAULT_MAX_HEAPS", ring_stream_wrapper::default_max_heaps)
        .def_readonly("DEFAULT_RING_HEAPS", ring_stream_wrapper::default_ring_heaps)
        .def_readonly("DEFAULT_UDP_MAX_SIZE", udp_reader::default_max_size)
        .def_readonly("DEFAULT_UDP_BUFFER_SIZE", udp_reader::default_buffer_size);
}

} // namespace recv
} // namespace spead2
