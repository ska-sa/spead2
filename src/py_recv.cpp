#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include <numpy/arrayobject.h>
#include <stdexcept>
#include "recv_udp.h"
#include "recv_mem.h"
#include "recv_receiver.h"
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
 * Stream that handles the magic necessary to reflect frozen heaps into
 * Python space and capture the reference to it.
 */
class ring_stream_wrapper : public ring_stream<ringbuffer_cond_gil<heap> >
{
public:
    using ring_stream::ring_stream;

    py::object pop()
    {
        frozen_heap fh = ring_stream::pop();
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
};

/**
 * Extends mem_reader to obtain data using the Python buffer protocol.
 *
 * The @ref buffer_view must be the first base class (rather than a member), so
 * that it is initialised before the arguments are passed to the @ref
 * mem_reader constructor.
 */
class buffer_reader : private buffer_view, public mem_reader
{
public:
    buffer_reader(boost::asio::io_service &io_service, stream &s, py::object obj)
        : buffer_view(obj),
        mem_reader(io_service, s, reinterpret_cast<const std::uint8_t *>(view.buf), view.len)
    {
    }
};

/**
 * Wraps @ref receiver to have add functions for each type of reader.
 */
class receiver_wrapper : public receiver
{
public:
    void add_buffer_reader(ring_stream_wrapper &s, py::object obj)
    {
        emplace_reader<buffer_reader>(s, obj);
    }

    /**
     * @todo add option for hostname to bind to, IPv4/v6, and sizes
     */
    void add_udp_reader(ring_stream_wrapper &s, int port)
    {
        boost::asio::ip::udp::endpoint endpoint(boost::asio::ip::address_v4::loopback(), port);
        emplace_reader<udp_reader>(s, endpoint);
    }

    void stop()
    {
        release_gil gil;
        receiver::stop();
    }
};

/* Wrapper to deal with import_array returning nothing in Python 2, NULL in
 * Python 3.
 */
#if PY_MAJOR_VERSION >= 3
static void *call_import_array(bool &success)
#else
static void call_import_array(bool &success)
#endif
{
    success = false;
    import_array(); // This is a macro that might return
    success = true;
#if PY_MAJOR_VERSION >= 3
    return NULL;
#endif
}

/// Register the receiver module with Boost.Python
void register_module()
{
    using namespace boost::python;
    using namespace spead::recv;

    // Needed to make NumPy functions work
    bool numpy_imported = false;
    call_import_array(numpy_imported);
    if (!numpy_imported)
        throw_error_already_set();

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
    class_<ring_stream_wrapper, boost::noncopyable>("Stream")
        .def(init<optional<bug_compat_mask, std::size_t> >())
        .def("__iter__", objects::identity_function())
        .def(
#if PY_MAJOR_VERSION >= 3
              // Python 3 uses __next__ for the iterator protocol
              "__next__"
#else
              "next"
#endif
        , &ring_stream_wrapper::pop);
    class_<receiver_wrapper, boost::noncopyable>("Receiver")
        .def("add_buffer_reader", &receiver_wrapper::add_buffer_reader, with_custodian_and_ward<1, 2>())
        .def("add_udp_reader", &receiver_wrapper::add_udp_reader, with_custodian_and_ward<1, 2>())
        .def("start", &receiver_wrapper::start)
        .def("stop", &receiver_wrapper::stop);
}

} // namespace recv
} // namespace spead
