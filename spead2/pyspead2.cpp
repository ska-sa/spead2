#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include <stdexcept>
#include "in.h"
#include "udp_in.h"
#include "mem_in.h"
#include "receiver.h"

namespace py = boost::python;

namespace spead
{
namespace in
{

class item_wrapper : public item
{
private:
    py::object owning_heap; // a wrapped frozen_heap

public:
    item_wrapper() = default;
    item_wrapper(const item &it, py::object owning_heap)
        : item(it), owning_heap(owning_heap) {}

    py::object get_value() const
    {
        return owning_heap;
    }
};

class frozen_heap_wrapper : public frozen_heap
{
private:
    PyObject *self;

public:
    frozen_heap_wrapper(PyObject *self, heap &h) : frozen_heap(std::move(h)), self(self) {}
    frozen_heap_wrapper(frozen_heap &&h) : frozen_heap(std::move(h)), self(nullptr) {}
    void set_self(PyObject *self) { this->self = self; }

    py::list get_items() const
    {
        py::object me(py::handle<>(py::borrowed(self)));
        std::vector<item> base = frozen_heap::get_items();
        py::list out;
        for (const item &it : base)
        {
            out.append(item_wrapper(it, me));
        }
        return out;
    }
};

class buffer_view : public boost::noncopyable
{
public:
    Py_buffer view;

    explicit buffer_view(py::object obj)
    {
        if (PyObject_GetBuffer(obj.ptr(), &view, PyBUF_SIMPLE) != 0)
            py::throw_error_already_set();
    }

    ~buffer_view()
    {
        PyBuffer_Release(&view);
    }
};

class heap_callback_wrapper
{
private:
    py::object obj;

public:
    typedef void result_type;

    heap_callback_wrapper(const py::object &obj) : obj(obj) {}

    void operator()(frozen_heap &&h)
    {
        frozen_heap_wrapper *wrapper = new frozen_heap_wrapper(std::move(h));
        std::unique_ptr<frozen_heap_wrapper> wrapper_ptr(wrapper);
        py::manage_new_object::apply<frozen_heap_wrapper *>::type converter;
        PyObject *arg_ptr = converter(wrapper);
        wrapper_ptr.release();
        py::object arg{py::handle<>(arg_ptr)};
        wrapper->set_self(arg_ptr);
        obj(arg);
    }
};

class buffer_stream : private buffer_view, public mem_stream
{
public:
    explicit buffer_stream(py::object obj)
        : buffer_view(obj),
        mem_stream(reinterpret_cast<const std::uint8_t *>(view.buf), view.len)
    {
    }

    void set_callback(const heap_callback_wrapper &callback)
    {
        stream::set_callback(callback);
    }
};

// Converter from Python object to std::function
struct heap_callback_from_callable
{
    static void *convertible(PyObject *obj)
    {
        if (PyCallable_Check(obj))
            return obj;
        else
            return NULL;
    }

    static void construct(
        PyObject *obj,
        py::converter::rvalue_from_python_stage1_data *data)
    {
        void *storage = (
            (py::converter::rvalue_from_python_storage<heap_callback_wrapper> *)
            data)->storage.bytes;

        py::object callback(py::handle<>(py::borrowed(obj)));
        new (storage) heap_callback_wrapper(callback);
        data->convertible = storage;
    }

    heap_callback_from_callable()
    {
        py::converter::registry::push_back(
            &convertible,
            &construct,
            py::type_id<heap_callback_wrapper>());
    }
};

} // namespace in
} // namespace spead

BOOST_PYTHON_MODULE(_spead2)
{
    using namespace boost::python;
    using namespace spead::in;

    heap_callback_from_callable();

    class_<heap, boost::noncopyable>("Heap", init<std::int64_t>());
    class_<frozen_heap, frozen_heap_wrapper, boost::noncopyable>("FrozenHeap", init<heap &>())
        .def("get_items", &frozen_heap_wrapper::get_items);
    class_<item_wrapper>("Item", no_init)
        .add_property("value", &item_wrapper::get_value);
    class_<buffer_stream, boost::noncopyable>("BufferStream", init<object>())
        .def("run", &buffer_stream::run)
        .def("set_callback", &buffer_stream::set_callback);
}
