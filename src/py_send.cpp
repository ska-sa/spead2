/**
 * @file
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL spead2_ARRAY_API
#define NO_IMPORT_ARRAY
#include <boost/python.hpp>
#include <stdexcept>
#include "send_heap.h"
#include "send_stream.h"

namespace py = boost::python;

namespace spead
{
namespace send
{

class heap_wrapper : public heap
{
public:
    using heap::heap;
};

/// Register the send module with Boost.Python
void register_module()
{
    using namespace boost::python;
    using namespace spead::send;

    // Create the module, and set it as the current boost::python scope so that
    // classes we define are added to this module rather than the root.
    py::object module(py::handle<>(py::borrowed(PyImport_AddModule("spead2._send"))));
    py::scope scope = module;

    class_<heap_wrapper, boost::noncopyable>("Heap", init<std::int64_t>())
        .def("add_descriptor", &heap_wrapper::add_descriptor);
}

} // namespace send
} // namespace spead
