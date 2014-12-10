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

namespace py = boost::python;

namespace spead
{

class release_gil
{
private:
    PyThreadState *save = nullptr;

public:
    release_gil()
    {
        release();
    }

    ~release_gil()
    {
        if (save != nullptr)
            PyEval_RestoreThread(save);
    }

    void release()
    {
        assert(save == nullptr);
        save = PyEval_SaveThread();
    }

    void acquire()
    {
        assert(save != nullptr);
        PyEval_RestoreThread(save);
        save = nullptr;
    }
};

namespace recv
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
        if (is_immediate)
        {
            return py::long_(value.immediate);
        }
        else
        {
            npy_intp dims[1];
            dims[0] = value.address.length;
            PyObject *obj_ptr = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, (void *) value.address.ptr);
            if (obj_ptr == NULL)
                py::throw_error_already_set();
            if (PyArray_SetBaseObject(
                    (PyArrayObject *) obj_ptr,
                    py::incref(owning_heap.ptr())) == -1)
            {
                py::decref(owning_heap.ptr());
                py::throw_error_already_set();
            }
            py::object obj{py::handle<>(obj_ptr)};
            return obj;
        }
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

// Wraps access to a Python buffer-protocol object
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

// Ringbuffer that releases the GIL while popping, and checks
// for KeyboardInterrupt
template<typename T>
class ringbuffer_wrapper : public ringbuffer<T>
{
public:
    using ringbuffer<T>::ringbuffer;

    T pop()
    {
        release_gil gil;
        std::unique_lock<std::mutex> lock(this->mutex);
        while (this->empty_unlocked())
        {
            this->data_cond.wait_for(lock, std::chrono::seconds(1));
            // Allow interpreter to catch KeyboardInterrupt. The timeout
            // ensures that we wake up periodically to check for this,
            // even if the signal doesn't cause spurious wakeup.
            gil.acquire();
            if (PyErr_CheckSignals() == -1)
                py::throw_error_already_set();
            gil.release();
        }
        return this->pop_unlocked();
    }
};

class ring_stream_wrapper : public ring_stream<ringbuffer_wrapper<heap> >
{
public:
    using ring_stream::ring_stream;

    py::object pop()
    {
        frozen_heap fh = ring_stream::pop();
        frozen_heap_wrapper *wrapper = new frozen_heap_wrapper(std::move(fh));
        std::unique_ptr<frozen_heap_wrapper> wrapper_ptr(wrapper);
        py::manage_new_object::apply<frozen_heap_wrapper *>::type converter;
        PyObject *obj_ptr = converter(wrapper);
        wrapper_ptr.release();
        py::object obj{py::handle<>(obj_ptr)};
        wrapper->set_self(obj_ptr);
        return obj;
    }
};

class buffer_reader : private buffer_view, public mem_reader
{
public:
    buffer_reader(stream *s, py::object obj)
        : buffer_view(obj),
        mem_reader(s, reinterpret_cast<const std::uint8_t *>(view.buf), view.len)
    {
    }
};

class receiver_wrapper : public receiver
{
public:
    void add_buffer_reader(ring_stream_wrapper *s, py::object obj)
    {
        emplace_reader<buffer_reader>(s, obj);
    }
};

} // namespace recv
} // namespace spead

BOOST_PYTHON_MODULE(_recv)
{
    using namespace boost::python;
    using namespace spead::recv;

    import_array();

    class_<frozen_heap, frozen_heap_wrapper, boost::noncopyable>("Heap", no_init)
        .add_property("cnt", &frozen_heap_wrapper::cnt)
        .def("get_items", &frozen_heap_wrapper::get_items);
    class_<item_wrapper>("Item", no_init)
        .def_readwrite("id", &item_wrapper::id)
        .add_property("value", &item_wrapper::get_value);
    class_<ring_stream_wrapper, boost::noncopyable>("Stream")
        .def(init<std::size_t>())
        .def("pop", &ring_stream_wrapper::pop);
    class_<receiver_wrapper, boost::noncopyable>("Receiver")
        .def("add_buffer_reader", &receiver_wrapper::add_buffer_reader, with_custodian_and_ward<1, 2>())
        .def("start", &receiver_wrapper::start)
        .def("stop", &receiver_wrapper::stop)
        .def("join", &receiver_wrapper::join);
}
