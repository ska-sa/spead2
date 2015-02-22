#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include <numpy/arrayobject.h>
#include <stdexcept>
#include "recv_udp.h"
#include "recv_mem.h"
#include "recv_stream.h"
#include "recv_ring_stream.h"
#include "recv_heap.h"
#include "recv_frozen_heap.h"
#include "py_common.h"

namespace py = boost::python;

namespace spead
{
namespace recv
{

/**
 * Wraps @ref item to provide safe memory management. The item references
 * memory inside the frozen heap, so it needs to hold a reference to that
 * heap, as do any memoryviews created on the value.
 */
class item_wrapper : public item
{
private:
    /// Python object containing a @ref frozen_heap
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
 * Wrapper for frozen heaps. For some reason boost::python doesn't provide
 * a way for object methods to retrieve their self object, so we have to store
 * a copy of it here, and set it whenever we build one of these objects. We
 * need this for @ref item_wrapper.
 */
class frozen_heap_wrapper : public frozen_heap
{
private:
    PyObject *self;

public:
    /// Constructor when built from Python
    frozen_heap_wrapper(frozen_heap &&h) : frozen_heap(std::move(h)), self(nullptr) {}
    void set_self(PyObject *self) { this->self = self; }

    /// Wrap @ref frozen_heap::get_items, and convert vector to a Python list
    py::list get_items() const
    {
        std::vector<item> base = frozen_heap::get_items();
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

    /// Wrap @ref frozen_heap::get_descriptors, and convert vector to a Python list
    py::list get_descriptors() const
    {
        std::vector<descriptor> descriptors = frozen_heap::get_descriptors();
        py::list out;
        for (const descriptor& d : descriptors)
            out.append(d);
        return out;
    }
};

/**
 * Extends mem_reader to obtain data using the Python buffer protocol.
 */
class buffer_reader : public mem_reader
{
private:
    std::unique_ptr<buffer_view> view;
public:
    explicit buffer_reader(stream &s, std::unique_ptr<buffer_view> &view)
        : mem_reader(s, reinterpret_cast<const std::uint8_t *>(view->view.buf), view->view.len),
        view(std::move(view))
    {
    }
};

/**
 * Stream that handles the magic necessary to reflect frozen heaps into
 * Python space and capture the reference to it.
 *
 * The GIL needs to be handled carefully. Any operation run by the thread pool
 * might need to take the GIL to do logging. Thus, any operation that blocks
 * on completion of code scheduled through the thread pool must drop the GIL
 * first.
 */
class ring_stream_wrapper : public ring_stream<ringbuffer_fd_gil<heap> >
{
private:
    py::object wrap_frozen_heap(frozen_heap &&fh)
    {
        // We need to allocate a new object to have type frozen_heap_wrapper,
        // but it can move all the resources out of fh.
        frozen_heap_wrapper *wrapper = new frozen_heap_wrapper(std::move(fh));
        std::unique_ptr<frozen_heap_wrapper> wrapper_ptr(wrapper);
        // Wrap the pointer up into a Python object
        py::manage_new_object::apply<frozen_heap_wrapper *>::type converter;
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
        return wrap_frozen_heap(ring_stream::pop());
    }

    py::object get_nowait()
    {
        return wrap_frozen_heap(ring_stream::try_pop());
    }

    int get_fd() const
    {
        return get_ringbuffer().get_fd();
    }

    void set_mem_pool(std::shared_ptr<mem_pool> pool)
    {
        release_gil gil;
        ring_stream::set_mem_pool(std::move(pool));
    }

    void add_buffer_reader(py::object obj)
    {
        std::unique_ptr<buffer_view> view{new buffer_view(obj)};
        release_gil gil;
        emplace_reader<buffer_reader>(std::ref(view));
    }

    void add_udp_reader(
        int port,
        std::size_t max_size = udp_reader::default_max_size,
        std::size_t default_buffer_size = udp_reader::default_buffer_size,
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
        emplace_reader<udp_reader>(endpoint, max_size, default_buffer_size);
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

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ring_stream_add_udp_reader_overloads, add_udp_reader, 1, 4)

/// Register the receiver module with Boost.Python
void register_module()
{
    using namespace boost::python;
    using namespace spead::recv;

    // Create the module, and set it as the current boost::python scope so that
    // classes we define are added to this module rather than the root.
    py::object module(py::handle<>(py::borrowed(PyImport_AddModule("spead2._recv"))));
    py::scope scope = module;

    class_<frozen_heap_wrapper, boost::noncopyable>("Heap", no_init)
        .add_property("cnt", &frozen_heap_wrapper::cnt)
        .add_property("bug_compat", &frozen_heap_wrapper::get_bug_compat)
        .def("get_items", &frozen_heap_wrapper::get_items)
        .def("get_descriptors", &frozen_heap_wrapper::get_descriptors);
    class_<item_wrapper>("RawItem", no_init)
        .def_readwrite("id", &item_wrapper::id)
        .add_property("value", &item_wrapper::get_value);
    class_<ring_stream_wrapper, boost::noncopyable>("Stream",
            init<thread_pool_wrapper &, optional<bug_compat_mask, std::size_t> >()[
                with_custodian_and_ward<1, 2>()])
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
        .def("set_mem_pool", &ring_stream_wrapper::set_mem_pool)
        .def("add_buffer_reader", &ring_stream_wrapper::add_buffer_reader)
        .def("add_udp_reader", &ring_stream_wrapper::add_udp_reader,
             ring_stream_add_udp_reader_overloads())
        .def("stop", &ring_stream_wrapper::stop)
        .add_property("fd", &ring_stream_wrapper::get_fd);
}

} // namespace recv
} // namespace spead
