#include <boost/python.hpp>
#include "py_common.h"
#include "common_ringbuffer.h"
#include "common_defines.h"

namespace spead
{

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
}

} // namespace spead

#include "py_recv.h"

BOOST_PYTHON_MODULE(_spead2)
{
    spead::register_module();
    spead::recv::register_module();
}
