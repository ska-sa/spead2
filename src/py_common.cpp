#include <boost/python.hpp>
#include <memory>
#include "py_common.h"
#include "common_ringbuffer.h"
#include "common_defines.h"
#include "common_logging.h"
#include "common_mempool.h"

namespace py = boost::python;

namespace spead
{

class log_function_python
{
private:
    py::object logger;
public:
    typedef void result_type;

    explicit log_function_python(const py::object &logger) : logger(logger) {}

    void operator()(log_level level, const std::string &msg)
    {
        acquire_gil gil;

        static const char *const level_methods[] =
        {
            "warning",
            "info",
            "debug"
        };
        unsigned int level_idx = static_cast<unsigned int>(level);
        assert(level_idx < sizeof(level_methods) / sizeof(level_methods[0]));
        logger.attr(level_methods[level_idx])("%s", msg);
    }
};

static PyObject *ringbuffer_stopped_type;
static PyObject *ringbuffer_empty_type;

static void translate_exception(const ringbuffer_stopped &e)
{
    PyErr_SetString(ringbuffer_stopped_type, e.what());
}

static void translate_exception(const ringbuffer_empty &e)
{
    PyErr_SetString(ringbuffer_empty_type, e.what());
}

static void translate_exception_stop_iteration(const stop_iteration &e)
{
    PyErr_SetString(PyExc_StopIteration, e.what());
}

static py::object descriptor_get_shape(const descriptor &d)
{
    py::list out;
    for (const auto &size : d.shape)
    {
        out.append(size);
    }
    return out;
}

static py::object descriptor_get_format(const descriptor &d)
{
    py::list out;
    for (const auto &item : d.format)
    {
        out.append(py::make_tuple(item.first, item.second));
    }
    return out;
}

static py::object int_to_object(long ival)
{
#if PY_MAJOR_VERSION >= 3
    PyObject *obj = PyLong_FromLong(ival);
#else
    PyObject *obj = PyInt_FromLong(ival);
#endif
    if (!obj)
        py::throw_error_already_set();
    return py::object(py::handle<>(obj));
}

template<typename T>
static void create_exception(PyObject *&type, const char *name, const char *basename)
{
    type = PyErr_NewException(const_cast<char *>(name), NULL, NULL);
    if (type == NULL)
        py::throw_error_already_set();
    py::scope().attr(basename) = py::handle<>(py::borrowed(type));
    py::register_exception_translator<T>((void (*)(const T &)) &translate_exception);
}

class mempool_wrapper
{
private:
    std::shared_ptr<mempool> pool;

public:
    template<typename... Args>
    mempool_wrapper(Args&&... args)
    : pool(std::make_shared<mempool>(args...))
    {
    }

    operator std::shared_ptr<mempool>()
    {
        return pool;
    }
};

static void register_module()
{
    using namespace boost::python;
    using namespace spead;

    create_exception<ringbuffer_stopped>(ringbuffer_stopped_type, "spead2.Stopped", "Stopped");
    create_exception<ringbuffer_empty>(ringbuffer_empty_type, "spead2.Empty", "Empty");
    register_exception_translator<stop_iteration>(&translate_exception_stop_iteration);

    py::setattr(scope(), "BUG_COMPAT_DESCRIPTOR_WIDTHS", int_to_object(BUG_COMPAT_DESCRIPTOR_WIDTHS));
    py::setattr(scope(), "BUG_COMPAT_SHAPE_BIT_1", int_to_object(BUG_COMPAT_SHAPE_BIT_1));
    py::setattr(scope(), "BUG_COMPAT_SWAP_ENDIAN", int_to_object(BUG_COMPAT_SWAP_ENDIAN));

    class_<mempool_wrapper>("Mempool", init<std::size_t, std::size_t, std::size_t, std::size_t>());

    // TODO: make shape and format read-write
    class_<descriptor>("RawDescriptor")
        .def_readwrite("id", &descriptor::id)
        .def_readwrite("name", &descriptor::name)
        .def_readwrite("description", &descriptor::description)
        .add_property("shape", &descriptor_get_shape)
        .add_property("format", &descriptor_get_format)
        .def_readwrite("numpy_header", &descriptor::numpy_header);

    object logging_module = import("logging");
    object logger = logging_module.attr("getLogger")("spead2");
    set_log_function(log_function_python(logger));
}

} // namespace spead

#include "py_recv.h"

BOOST_PYTHON_MODULE(_spead2)
{
    spead::register_module();
    spead::recv::register_module();
}
