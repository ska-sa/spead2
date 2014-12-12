#include <boost/python.hpp>
#include "py_common.h"
#include "common_ringbuffer.h"
#include "common_defines.h"
#include "common_logging.h"

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

/// Converts the exception into a Python @c StopIteration
static void translate_ringbuffer_stopped(const ringbuffer_stopped &e)
{
    PyErr_SetString(PyExc_StopIteration, e.what());
}

static void register_module()
{
    using namespace boost::python;
    using namespace spead;

    register_exception_translator<ringbuffer_stopped>(&translate_ringbuffer_stopped);

    // TODO: missing shape and format
    class_<descriptor>("RawDescriptor")
        .def_readwrite("id", &descriptor::id)
        .def_readwrite("name", &descriptor::name)
        .def_readwrite("description", &descriptor::description)
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
