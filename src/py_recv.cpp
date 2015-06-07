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
#include "recv_udp.h"
#include "recv_mem.h"
#include "recv_stream.h"
#include "recv_ring_stream.h"
#include "recv_live_heap.h"
#include "recv_heap.h"
#include "py_common.h"

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
    py::object owning_heap;

public:
    item_wrapper() = default;
    item_wrapper(const item &it, PyObject *owning_heap)
        : item(it), owning_heap(py::handle<>(py::borrowed(owning_heap))) {}

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
                py::incref(owning_heap.ptr())) == -1)
        {
            py::decref(owning_heap.ptr());
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
 * Wrapper for heaps. For some reason boost::python doesn't provide
 * a way for object methods to retrieve their self object, so we have to store
 * a copy of it here, and set it whenever we build one of these objects. We
 * need this for @ref item_wrapper.
 */
class heap_wrapper : public heap
{
private:
    PyObject *self;

public:
    /// Constructor when built from Python
    heap_wrapper(heap &&h) : heap(std::move(h)), self(nullptr) {}
    void set_self(PyObject *self) { this->self = self; }

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

    /**
     * Return the flavour by value instead of by const reference. In theory
     * this should be achievable with a call policy, but it seems to lead to
     * runtime errors in overload resolution.
     */
    flavour get_flavour() const { return heap::get_flavour(); }
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
class ring_stream_wrapper : public ring_stream<ringbuffer_semaphore<live_heap, semaphore_gil> >
{
private:
    /// Holds a Python reference to the thread pool
    py::handle<> thread_pool_handle;
    friend void register_module();

    py::object wrap_heap(heap &&fh)
    {
        // We need to allocate a new object to have type heap_wrapper,
        // but it can move all the resources out of fh.
        heap_wrapper *wrapper = new heap_wrapper(std::move(fh));
        std::unique_ptr<heap_wrapper> wrapper_ptr(wrapper);
        // Wrap the pointer up into a Python object
        py::manage_new_object::apply<heap_wrapper *>::type converter;
        PyObject *obj_ptr = converter(wrapper);
        wrapper_ptr.release();
        py::object obj{py::handle<>(obj_ptr)};
        // Tell the wrapper what its Python handle is
        wrapper->set_self(obj_ptr);
        return obj;
    }

public:
    using ring_stream::ring_stream;

    py::object next()
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

    py::object get()
    {
        return wrap_heap(ring_stream::pop());
    }

    py::object get_nowait()
    {
        return wrap_heap(ring_stream::try_pop());
    }

    int get_fd() const
    {
        return get_ringbuffer().get_fd();
    }

    void set_memory_pool(std::shared_ptr<memory_pool> pool)
    {
        release_gil gil;
        ring_stream::set_memory_pool(std::move(pool));
    }

    void add_buffer_reader(py::object buffer)
    {
        buffer_view view(buffer);
        release_gil gil;
        emplace_reader<buffer_reader>(std::ref(view));
    }

    void add_udp_reader(
        int port,
        std::size_t max_size = udp_reader::default_max_size,
        std::size_t buffer_size = udp_reader::default_buffer_size,
        const std::string &bind_hostname = "")
    {
        using boost::asio::ip::udp;
        release_gil gil;
        udp::endpoint endpoint(boost::asio::ip::address_v4::any(), port);
        if (!bind_hostname.empty())
        {
            udp::resolver resolver(get_strand().get_io_service());
            udp::resolver::query query(bind_hostname, "", udp::resolver::query::passive | udp::resolver::query::address_configured);
            endpoint.address(resolver.resolve(query)->endpoint().address());
        }
        emplace_reader<udp_reader>(endpoint, max_size, buffer_size);
    }

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

    class_<heap_wrapper, boost::noncopyable>("Heap", no_init)
        .add_property("cnt", &heap_wrapper::get_cnt)
        .add_property("flavour", &heap_wrapper::get_flavour)
        .def("get_items", &heap_wrapper::get_items)
        .def("get_descriptors", &heap_wrapper::get_descriptors);
    class_<item_wrapper>("RawItem", no_init)
        .def_readonly("id", &item_wrapper::id)
        .def_readonly("is_immediate", &item_wrapper::is_immediate)
        .add_property("value", &item_wrapper::get_value);
    class_<ring_stream_wrapper, boost::noncopyable>("Stream",
            init<thread_pool_wrapper &, bug_compat_mask, std::size_t>(
                (arg("thread_pool"), arg("bug_compat") = 0,
                 arg("max_heaps") = ring_stream_wrapper::default_max_heaps))[
                store_handle_postcall<ring_stream_wrapper, &ring_stream_wrapper::thread_pool_handle, 1, 2>()])
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
        .def("set_memory_pool", &ring_stream_wrapper::set_memory_pool,
             arg("pool"))
        .def("add_buffer_reader", &ring_stream_wrapper::add_buffer_reader,
             arg("buffer"))
        .def("add_udp_reader", &ring_stream_wrapper::add_udp_reader,
             (arg("port"),
              arg("max_size") = udp_reader::default_max_size,
              arg("buffer_size") = udp_reader::default_buffer_size,
              arg("bind_hostname") = std::string()))
        .def("stop", &ring_stream_wrapper::stop)
        .add_property("fd", &ring_stream_wrapper::get_fd)
        .def_readonly("DEFAULT_MAX_HEAPS", ring_stream_wrapper::default_max_heaps)
        .def_readonly("DEFAULT_UDP_MAX_SIZE", udp_reader::default_max_size)
        .def_readonly("DEFAULT_UDP_BUFFER_SIZE", udp_reader::default_buffer_size);
}

} // namespace recv
} // namespace spead2
