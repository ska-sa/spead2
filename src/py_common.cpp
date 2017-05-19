/* Copyright 2015, 2017 SKA South Africa
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

/**
 * @file
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <boost/system/system_error.hpp>
#include <memory>
#include <spead2/py_common.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_defines.h>
#include <spead2/common_flavour.h>
#include <spead2/common_logging.h>
#include <spead2/common_memory_pool.h>
#include <spead2/common_thread_pool.h>

namespace py = pybind11;

namespace spead2
{

static void translate_exception_boost_io_error(std::exception_ptr p)
{
    try
    {
        if (p)
            std::rethrow_exception(p);
    }
    catch (const boost_io_error &e)
    {
        py::tuple args = py::make_tuple(e.code().value(), e.what());
        PyErr_SetObject(PyExc_IOError, args.ptr());
    }
}

thread_pool_wrapper::~thread_pool_wrapper()
{
    stop();
}

void thread_pool_wrapper::stop()
{
    py::gil_scoped_release gil;
    thread_pool::stop();
}

py::buffer_info request_buffer_info(py::buffer &buffer, int extra_flags)
{
    std::unique_ptr<Py_buffer> view(new Py_buffer);
    int flags = PyBUF_STRIDES | PyBUF_FORMAT | extra_flags;
    if (PyObject_GetBuffer(buffer.ptr(), view.get(), flags) != 0)
        throw py::error_already_set();
    py::buffer_info info(view.get());
    view.release();
    return info;
}

py::module register_module()
{
    using namespace spead2;
    using namespace pybind11::literals;

    py::module m("spead2._spead2");
    py::register_exception<ringbuffer_stopped>(m, "Stopped");
    py::register_exception<ringbuffer_empty>(m, "Empty");
    py::register_exception_translator(translate_exception_boost_io_error);

#define EXPORT_ENUM(x) (m.attr(#x) = long(x))
    EXPORT_ENUM(BUG_COMPAT_DESCRIPTOR_WIDTHS);
    EXPORT_ENUM(BUG_COMPAT_SHAPE_BIT_1);
    EXPORT_ENUM(BUG_COMPAT_SWAP_ENDIAN);
    EXPORT_ENUM(BUG_COMPAT_PYSPEAD_0_5_2);

    EXPORT_ENUM(NULL_ID);
    EXPORT_ENUM(HEAP_CNT_ID);
    EXPORT_ENUM(HEAP_LENGTH_ID);
    EXPORT_ENUM(PAYLOAD_OFFSET_ID);
    EXPORT_ENUM(PAYLOAD_LENGTH_ID);
    EXPORT_ENUM(DESCRIPTOR_ID);
    EXPORT_ENUM(STREAM_CTRL_ID);

    EXPORT_ENUM(DESCRIPTOR_NAME_ID);
    EXPORT_ENUM(DESCRIPTOR_DESCRIPTION_ID);
    EXPORT_ENUM(DESCRIPTOR_SHAPE_ID);
    EXPORT_ENUM(DESCRIPTOR_FORMAT_ID);
    EXPORT_ENUM(DESCRIPTOR_ID_ID);
    EXPORT_ENUM(DESCRIPTOR_DTYPE_ID);

    EXPORT_ENUM(CTRL_STREAM_START);
    EXPORT_ENUM(CTRL_DESCRIPTOR_REISSUE);
    EXPORT_ENUM(CTRL_STREAM_STOP);
    EXPORT_ENUM(CTRL_DESCRIPTOR_UPDATE);

    EXPORT_ENUM(MEMCPY_STD);
    EXPORT_ENUM(MEMCPY_NONTEMPORAL);
#undef EXPORT_ENUM

    spead2::class_<flavour>(m, "Flavour")
        .def(py::init<int, int, int, bug_compat_mask>(),
             "version"_a, "item_pointer_bits"_a,
             "heap_address_bits"_a, "bug_compat"_a=0)
        .def(py::init<>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def_property_readonly("version", SPEAD2_PTMF(flavour, get_version))
        .def_property_readonly("item_pointer_bits", SPEAD2_PTMF(flavour, get_item_pointer_bits))
        .def_property_readonly("heap_address_bits", SPEAD2_PTMF(flavour, get_heap_address_bits))
        .def_property_readonly("bug_compat", SPEAD2_PTMF(flavour, get_bug_compat));

    spead2::class_<memory_allocator, std::shared_ptr<memory_allocator>>(m, "MemoryAllocator")
        .def(py::init<>());

    spead2::class_<mmap_allocator, memory_allocator, std::shared_ptr<mmap_allocator>>(
        m, "MmapAllocator")
        .def(py::init<int>(), "flags"_a=0);

    spead2::class_<memory_pool, memory_allocator, std::shared_ptr<memory_pool>>(
        m, "MemoryPool")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, std::shared_ptr<memory_allocator>>(),
             "lower"_a, "upper"_a, "max_free"_a, "initial"_a, py::arg_v("allocator", nullptr, "None"))
        .def(py::init<std::shared_ptr<thread_pool>, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::shared_ptr<memory_allocator>>(),
             "thread_pool"_a, "lower"_a, "upper"_a, "max_free"_a, "initial"_a, "low_water"_a, "allocator"_a);

    spead2::class_<thread_pool_wrapper, std::shared_ptr<thread_pool_wrapper>>(m, "ThreadPool")
        .def(py::init<int>(), "threads"_a = 1)
        .def(py::init<int, const std::vector<int> &>(), "threads"_a, "affinity"_a)
        .def_static("set_affinity", &thread_pool_wrapper::set_affinity)
        .def("stop", SPEAD2_PTMF(thread_pool_wrapper, stop));

    spead2::class_<descriptor>(m, "RawDescriptor")
        .def(py::init<>())
        .def_readwrite("id", &descriptor::id)
        .def_property("name", bytes_getter(&descriptor::name), bytes_setter(&descriptor::name))
        .def_property("description", bytes_getter(&descriptor::description), bytes_setter(&descriptor::description))
        .def_property("shape", [](const descriptor &d) -> py::list
        {
            py::list out;
            for (const auto &size : d.shape)
            {
                if (size >= 0)
                    out.append(size);
                else
                    out.append(py::none());
            }
            return out;
        }, [](descriptor &d, py::sequence shape)
        {
            std::vector<std::int64_t> out;
            out.reserve(len(shape));
            for (std::size_t i = 0; i < len(shape); i++)
            {
                py::object value = shape[i];
                if (value.is_none())
                    out.push_back(-1);
                else
                {
                    std::int64_t v = value.cast<std::int64_t>();
                    // TODO: verify range (particularly, >= 0)
                    out.push_back(v);
                }
            }
            d.shape = std::move(out);
        })
        .def_readwrite("format", &descriptor::format)
        .def_property("numpy_header", bytes_getter(&descriptor::numpy_header), bytes_setter(&descriptor::numpy_header))
    ;

    py::object logging_module = py::module::import("logging");
    py::object logger = logging_module.attr("getLogger")("spead2");
    set_log_function(log_function_python(logger));

    return m;
}

} // namespace spead2
