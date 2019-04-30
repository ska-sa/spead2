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
#include <list>
#include <functional>
#include <spead2/py_common.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_defines.h>
#include <spead2/common_flavour.h>
#include <spead2/common_logging.h>
#include <spead2/common_memory_pool.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_inproc.h>
#if SPEAD2_USE_IBV
# include <spead2/common_ibv.h>
#endif

namespace py = pybind11;

namespace spead2
{

namespace detail
{

static std::list<std::function<void()>> stop_entries;
static std::function<void(log_level, const std::string &)> orig_logger;
static std::unique_ptr<log_function_python> our_logger;

static void run_exit_stoppers()
{
    while (!stop_entries.empty())
        stop_entries.front()();
    // Clear up our custom logger
    set_log_function(orig_logger);
    our_logger.reset();
}

} // namespace detail

exit_stopper::exit_stopper(std::function<void()> callback)
    : entry(detail::stop_entries.insert(detail::stop_entries.begin(), std::move(callback)))
{
}

void exit_stopper::reset()
{
    if (entry != detail::stop_entries.end())
    {
        detail::stop_entries.erase(entry);
        entry = detail::stop_entries.end();
    }
}

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

template class socket_wrapper<boost::asio::ip::udp::socket>;
template class socket_wrapper<boost::asio::ip::tcp::socket>;
template class socket_wrapper<boost::asio::ip::tcp::acceptor>;

boost::asio::ip::address make_address_no_release(
    boost::asio::io_service &io_service, const std::string &hostname,
    boost::asio::ip::resolver_query_base::flags flags)
{
    if (hostname == "")
        return boost::asio::ip::address();
    using boost::asio::ip::udp;
    udp::resolver resolver(io_service);
    udp::resolver::query query(hostname, "", flags);
    return resolver.resolve(query)->endpoint().address();
}

void deprecation_warning(const char *msg)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning, msg, 1) == -1)
        throw py::error_already_set();
}

thread_pool_wrapper::~thread_pool_wrapper()
{
    stop();
}

void thread_pool_wrapper::stop()
{
    stopper.reset();
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

constexpr unsigned int log_function_python::num_levels;
const char *const log_function_python::level_methods[log_function_python::num_levels] =
{
    "warning",
    "info",
    "debug"
};

log_function_python::log_function_python(
    pybind11::object logger, std::size_t ring_size) :
        overflowed(false),
        ring(ring_size)
{
    for (unsigned int i = 0; i < num_levels; i++)
        log_methods[i] = logger.attr(level_methods[i]);
    thread = std::thread([this] () { run(); });
}

void log_function_python::run()
{
    try
    {
        while (true)
        {
            auto msg = ring.pop();
            py::gil_scoped_acquire gil;
            log(msg.first, msg.second);
            /* If there are multiple messages queued, consume them while
             * the GIL is held, rather than dropping and regaining the
             * GIL; but limit it, so that we don't starve other threads
             * of the GIL.
             */
            try
            {
                for (int pass = 1; pass < 1024; pass++)
                {
                    msg = ring.try_pop();
                    log(msg.first, msg.second);
                }
            }
            catch (ringbuffer_empty &)
            {
            }
            if (overflowed.exchange(false))
                log(log_level::warning,
                    "Log ringbuffer was full - some log messages were dropped");
        }
    }
    catch (ringbuffer_stopped &)
    {
        // Could possibly report the overflowed flag here again - but this may be
        // deep into interpreter shutdown and it might not be safe to log.
    }
    catch (std::exception &e)
    {
        std::cerr << "Logger thread crashed with exception " << e.what() << '\n';
    }
}

void log_function_python::log(log_level level, const std::string &msg) const
{
    try
    {
        unsigned int level_idx = static_cast<unsigned int>(level);
        assert(level_idx < num_levels);
        log_methods[level_idx]("%s", msg);
    }
    catch (py::error_already_set &e)
    {
        // This can happen during interpreter shutdown, because the modules
        // needed for the logging have already been unloaded.
    }
}

void log_function_python::operator()(log_level level, const std::string &msg)
{
    /* A blocking push can potentially lead to deadlock: the consumer may be
     * blocked waiting for the GIL, and possibly we may be holding the GIL.
     * If there is so much logging that the consumer can't keep up, we
     * probably want to throttle the log messages anyway, so we just set a
     * flag.
     */
    try
    {
        ring.try_emplace(level, msg);
    }
    catch (ringbuffer_full &)
    {
        overflowed = true;
    }
}

void log_function_python::stop()
{
    stopper.reset();
    {
        py::gil_scoped_release gil;
        ring.stop();
        if (thread.joinable())
            thread.join();
    }
    for (unsigned int i = 0; i < num_levels; i++)
        log_methods[i] = py::object();
}

void register_module(py::module m)
{
    using namespace spead2;
    using namespace pybind11::literals;

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

    m.def("log_info", [](const std::string &msg) { log_info("%s", msg); },
          "Log a message at INFO level (for testing only)");

    py::class_<flavour>(m, "Flavour")
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

    py::class_<memory_allocator, std::shared_ptr<memory_allocator>>(m, "MemoryAllocator")
        .def(py::init<>());

    py::class_<mmap_allocator, memory_allocator, std::shared_ptr<mmap_allocator>>(
        m, "MmapAllocator")
        .def(py::init<int>(), "flags"_a=0);

    py::class_<memory_pool, memory_allocator, std::shared_ptr<memory_pool>>(
        m, "MemoryPool")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, std::shared_ptr<memory_allocator>>(),
             "lower"_a, "upper"_a, "max_free"_a, "initial"_a, py::arg_v("allocator", nullptr, "None"))
        .def(py::init<std::shared_ptr<thread_pool>, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::shared_ptr<memory_allocator>>(),
             "thread_pool"_a, "lower"_a, "upper"_a, "max_free"_a, "initial"_a, "low_water"_a, "allocator"_a)
        .def_property("warn_on_empty",
                      &memory_pool::get_warn_on_empty, &memory_pool::set_warn_on_empty);

    py::class_<thread_pool_wrapper, std::shared_ptr<thread_pool_wrapper>>(m, "ThreadPool")
        .def(py::init<int>(), "threads"_a = 1)
        .def(py::init<int, const std::vector<int> &>(), "threads"_a, "affinity"_a)
        .def_static("set_affinity", &thread_pool_wrapper::set_affinity)
        .def("stop", SPEAD2_PTMF(thread_pool_wrapper, stop));

    py::class_<inproc_queue, std::shared_ptr<inproc_queue>>(m, "InprocQueue")
        .def(py::init<>())
        .def("stop", SPEAD2_PTMF(inproc_queue, stop));

    py::class_<descriptor>(m, "RawDescriptor")
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
#if SPEAD2_USE_IBV
    py::class_<ibv_context_t>(m, "IbvContext")
        .def(py::init([](const std::string &interface_address)
            {
                py::gil_scoped_release release;
                boost::asio::io_service io_service;
                return ibv_context_t(make_address_no_release(
                    io_service, interface_address, boost::asio::ip::udp::resolver::query::passive));
            }), "interface"_a)
        .def("reset", [](ibv_context_t &self) { self.reset(); })
    ;
#endif
}

void register_logging()
{
    py::object logging_module = py::module::import("logging");
    py::object logger = logging_module.attr("getLogger")("spead2");
    detail::our_logger.reset(new log_function_python(logger));
    detail::orig_logger = set_log_function(std::ref(*detail::our_logger));
}

void register_atexit()
{
    py::module atexit_mod = py::module::import("atexit");
    atexit_mod.attr("register")(py::cpp_function(detail::run_exit_stoppers));
}

} // namespace spead2
