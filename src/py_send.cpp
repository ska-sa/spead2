/* Copyright 2015 SKA South Africa
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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL spead2_ARRAY_API
#define NO_IMPORT_ARRAY
#include <boost/python.hpp>
#include <boost/system/system_error.hpp>
#include <stdexcept>
#include <mutex>
#include <utility>
#include <memory>
#include <unistd.h>
#include <spead2/send_heap.h>
#include <spead2/send_stream.h>
#include <spead2/send_udp.h>
#include <spead2/send_udp_ibv.h>
#include <spead2/send_streambuf.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_semaphore.h>
#include <spead2/py_common.h>

namespace py = boost::python;

namespace spead2
{
namespace send
{

class heap_wrapper : public heap
{
private:
    std::vector<buffer_view> item_buffers;

public:
    using heap::heap;
    void add_item(py::object item);
    void add_descriptor(py::object descriptor);
    flavour get_flavour() const;
};

void heap_wrapper::add_item(py::object item)
{
    std::int64_t id = py::extract<std::int64_t>(item.attr("id"));
    py::object buffer = item.attr("to_buffer")();
    bool allow_immediate = py::extract<bool>(item.attr("allow_immediate")());
    item_buffers.emplace_back(buffer);
    const auto &view = item_buffers.back().view;
    heap::add_item(id, view.buf, view.len, allow_immediate);
}

void heap_wrapper::add_descriptor(py::object object)
{
    heap::add_descriptor(py::extract<descriptor>(object.attr("to_raw")(heap::get_flavour())));
}

flavour heap_wrapper::get_flavour() const
{
    return heap::get_flavour();
}

class packet_generator_wrapper : public packet_generator
{
private:
    boost::python::handle<> heap_handle;
    friend void register_module();

public:
    using packet_generator::packet_generator;

    bytestring next();
};

bytestring packet_generator_wrapper::next()
{
    packet pkt = next_packet();
    if (pkt.buffers.empty())
        throw stop_iteration();
    return bytestring(boost::asio::buffers_begin(pkt.buffers),
                      boost::asio::buffers_end(pkt.buffers));
}

template<typename Base>
class stream_wrapper : public Base
{
private:
    struct callback_state
    {
        /**
         * Semaphore triggered by the callback. It uses a semaphore rather
         * than a promise because a semaphore can be interrupted.
         */
        semaphore_gil<semaphore> sem;
        /**
         * Error code from the callback.
         */
        boost::system::error_code ec;
        /**
         * Bytes transferred (encoded heap size).
         */
        item_pointer_t bytes_transferred = 0;
    };

public:
    using Base::Base;

    /// Sends heap synchronously
    item_pointer_t send_heap(const heap_wrapper &h, s_item_pointer_t cnt = -1)
    {
        /* The semaphore state needs to be in shared_ptr because if we are
         * interrupted and throw an exception, it still needs to exist until
         * the heap is sent.
         */
        auto state = std::make_shared<callback_state>();
        Base::async_send_heap(h, [state] (const boost::system::error_code &ec, item_pointer_t bytes_transferred)
        {
            state->ec = ec;
            state->bytes_transferred = bytes_transferred;
            state->sem.put();
        }, cnt);
        semaphore_get(state->sem);
        if (state->ec)
            throw boost_io_error(state->ec);
        else
            return state->bytes_transferred;
    }
};

template<typename Base>
class asyncio_stream_wrapper : public Base
{
private:
    struct callback_item
    {
        PyObject *callback;
        PyObject *h;  // heap: kept here because it can only be freed with the GIL
        boost::system::error_code ec;
        item_pointer_t bytes_transferred;
    };

    semaphore_gil<semaphore_fd> sem;
    std::vector<callback_item> callbacks;
    std::mutex callbacks_mutex;

    // Prevent copying: the callbacks vector cannot sanely be copied
    asyncio_stream_wrapper(const asyncio_stream_wrapper &) = delete;
    asyncio_stream_wrapper &operator=(const asyncio_stream_wrapper &) = delete;
public:
    using Base::Base;

    int get_fd() const { return sem.get_fd(); }

    bool async_send_heap(py::object h, py::object callback, s_item_pointer_t cnt = -1)
    {
        py::extract<heap_wrapper &> h2(h);
        /* Normally the callback should not refer to this, since it could have
         * been reaped by the time the callback occurs. We rely on Python to
         * hang on to a reference to self.
         *
         * The callback and heap are passed around by raw reference, because
         * it is not safe to use incref/decref operations without the GIL, and
         * attempting to use py::object instead of PyObject* tends to cause
         * these operations to occur without it being obvious.
         */
        PyObject *h_ptr = h.ptr();
        PyObject *callback_ptr = callback.ptr();
        Py_INCREF(h_ptr);
        Py_INCREF(callback_ptr);
        return Base::async_send_heap(h2(), [this, callback_ptr, h_ptr] (
            const boost::system::error_code &ec, item_pointer_t bytes_transferred)
        {
            bool was_empty;
            {
                std::unique_lock<std::mutex> lock(callbacks_mutex);
                was_empty = callbacks.empty();
                callbacks.push_back(callback_item{callback_ptr, h_ptr, ec, bytes_transferred});
            }
            if (was_empty)
                sem.put();
        }, cnt);
    }

    void process_callbacks()
    {
        sem.get();
        std::vector<callback_item> current_callbacks;
        {
            std::unique_lock<std::mutex> lock(callbacks_mutex);
            current_callbacks.swap(callbacks);
        }
        try
        {
            for (callback_item &item : current_callbacks)
            {
                Py_DECREF(item.h);
                item.h = NULL;
                py::object callback{py::handle<>(item.callback)};
                item.callback = NULL;
                py::object exc;
                if (item.ec)
                {
                    py::object exc_class(py::handle<>(py::borrowed(PyExc_IOError)));
                    exc = exc_class(item.ec.value(), item.ec.message());
                }
                callback(exc, item.bytes_transferred);
                // Ref to callback will be dropped in destructor for item
            }
        }
        catch (std::exception &e)
        {
            /* Clean up the remaining handles. Note that we only get here if
             * things have gone very badly wrong, such as an out-of-memory
             * error.
             */
            for (const callback_item &item : current_callbacks)
            {
                Py_XDECREF(item.h);
                Py_XDECREF(item.callback);
            }
            throw;
        }
    }

    ~asyncio_stream_wrapper()
    {
        for (const callback_item &item : callbacks)
        {
            Py_DECREF(item.h);
            Py_DECREF(item.callback);
        }
    }
};

static boost::asio::ip::address make_address(
    boost::asio::io_service &io_service, const std::string &hostname)
{
    release_gil gil;

    using boost::asio::ip::udp;
    udp::resolver resolver(io_service);
    udp::resolver::query query(hostname, "", boost::asio::ip::resolver_query_base::flags(0));
    return resolver.resolve(query)->endpoint().address();
}

static boost::asio::ip::udp::endpoint make_endpoint(
    boost::asio::io_service &io_service, const std::string &hostname, std::uint16_t port)
{
    return boost::asio::ip::udp::endpoint(make_address(io_service, hostname), port);
}

static boost::asio::ip::udp::socket make_socket(
    boost::asio::io_service &io_service, const boost::asio::ip::udp &protocol,
    const py::object &socket)
{
    using boost::asio::ip::udp;
    if (!socket.is_none())
    {
        int fd = py::extract<int>(socket.attr("fileno")());
        /* Need to duplicate the FD, since Python still owns the original */
        int fd2 = ::dup(fd);
        if (fd2 == -1)
        {
            PyErr_SetFromErrno(PyExc_OSError);
            throw py::error_already_set();
        }
        /* TODO: will this leak the FD if the constructor fails? Can the
         * constructor fail or is it just setting an FD in an object?
         */
        return boost::asio::ip::udp::socket(io_service, protocol, fd2);
    }
    else
        return boost::asio::ip::udp::socket(io_service, protocol);
}

template<typename Base>
class udp_stream_wrapper : public thread_pool_handle_wrapper, public Base
{
private:
    /* Intermediate chained constructors that has the hostname and port
     * converted to an endpoint, so that it can be used in turn to
     * construct the asio socket.
     */
    udp_stream_wrapper(
        thread_pool &pool,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        std::size_t buffer_size,
        const py::object &socket)
        : Base(
            make_socket(pool.get_io_service(), endpoint.protocol(), socket),
            endpoint,
            config, buffer_size)
    {
    }

    udp_stream_wrapper(
        thread_pool &pool,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl)
        : Base(pool.get_io_service(), endpoint, config, buffer_size, ttl)
    {
    }

    template<typename T>
    udp_stream_wrapper(
        thread_pool &pool,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        const T &interface)
        : Base(pool.get_io_service(), endpoint, config, buffer_size, ttl, interface)
    {
    }

public:
    udp_stream_wrapper(
        thread_pool &pool,
        const std::string &hostname,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        const py::object &socket)
        : udp_stream_wrapper(
            pool, make_endpoint(pool.get_io_service(), hostname, port),
            config, buffer_size, socket)
    {
    }

    udp_stream_wrapper(
        thread_pool &pool,
        const std::string &multicast_group,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl)
        : udp_stream_wrapper(
            pool, make_endpoint(pool.get_io_service(), multicast_group, port),
            config, buffer_size, ttl)
    {
    }

    udp_stream_wrapper(
        thread_pool &pool,
        const std::string &multicast_group,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        const std::string &interface_address)
        : udp_stream_wrapper(
            pool, make_endpoint(pool.get_io_service(), multicast_group, port),
            config, buffer_size, ttl,
            interface_address.empty() ?
                boost::asio::ip::address_v4::any() :
                make_address(pool.get_io_service(), interface_address))
    {
    }

    udp_stream_wrapper(
        thread_pool &pool,
        const std::string &multicast_group,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        unsigned int interface_index)
        : udp_stream_wrapper(
            pool, make_endpoint(pool.get_io_service(), multicast_group, port),
            config, buffer_size, ttl, interface_index)
    {
    }
};

#if SPEAD2_USE_IBV
template<typename Base>
class udp_ibv_stream_wrapper : public thread_pool_handle_wrapper, public Base
{
public:
    udp_ibv_stream_wrapper(
        thread_pool &pool,
        const std::string &multicast_group,
        std::uint16_t port,
        const stream_config &config,
        const std::string &interface_address,
        std::size_t buffer_size,
        int ttl,
        int comp_vector,
        int max_poll)
        : Base(pool.get_io_service(),
               make_endpoint(pool.get_io_service(), multicast_group, port),
               config,
               make_address(pool.get_io_service(), interface_address),
               buffer_size, ttl, comp_vector, max_poll)
    {
    }
};
#endif

class bytes_stream : private std::stringbuf, public thread_pool_handle_wrapper, public stream_wrapper<streambuf_stream>
{
public:
    bytes_stream(thread_pool &pool, const stream_config &config = stream_config())
        : stream_wrapper<streambuf_stream>(pool.get_io_service(), *this, config)
    {
    }

    bytestring getvalue() const
    {
        return str();
    }
};

template<typename T>
static boost::python::class_<T, boost::noncopyable> udp_stream_register(const char *name)
{
    using namespace boost::python;
    return class_<T, boost::noncopyable>(name, init<
            thread_pool_wrapper &, std::string, int, const stream_config &, std::size_t, const py::object &>(
                (arg("thread_pool"), arg("hostname"), arg("port"),
                 arg("config") = stream_config(),
                 arg("buffer_size") = T::default_buffer_size,
                 arg("socket") = py::object()))[
            store_handle_postcall<T, thread_pool_handle_wrapper, &thread_pool_handle_wrapper::thread_pool_handle, 1, 2>()])
        .def(init<thread_pool_wrapper &, std::string, int, const stream_config &, std::size_t, int>(
                (arg("thread_pool"), arg("multicast_group"), arg("port"),
                 arg("config") = stream_config(),
                 arg("buffer_size") = T::default_buffer_size,
                 arg("ttl")))[
            store_handle_postcall<T, thread_pool_handle_wrapper, &thread_pool_handle_wrapper::thread_pool_handle, 1, 2>()])
        .def(init<thread_pool_wrapper &, std::string, int, const stream_config &, std::size_t, int, std::string>(
                (arg("thread_pool"), arg("multicast_group"), arg("port"),
                 arg("config") = stream_config(),
                 arg("buffer_size") = T::default_buffer_size,
                 arg("ttl"),
                 arg("interface_address")))[
            store_handle_postcall<T, thread_pool_handle_wrapper, &thread_pool_handle_wrapper::thread_pool_handle, 1, 2>()])
        .def(init<thread_pool_wrapper &, std::string, int, const stream_config &, std::size_t, int, unsigned int>(
                (arg("thread_pool"), arg("multicast_group"), arg("port"),
                 arg("config") = stream_config(),
                 arg("buffer_size") = T::default_buffer_size,
                 arg("ttl"),
                 arg("interface_index")))[
            store_handle_postcall<T, thread_pool_handle_wrapper, &thread_pool_handle_wrapper::thread_pool_handle, 1, 2>()])
        .def_readonly("DEFAULT_BUFFER_SIZE", T::default_buffer_size);
}

#if SPEAD2_USE_IBV
template<typename T>
static boost::python::class_<T, boost::noncopyable> udp_ibv_stream_register(const char *name)
{
    using namespace boost::python;
    return class_<T, boost::noncopyable>(
        name,
        init<thread_pool_wrapper &, std::string, int, const stream_config &, std::string, std::size_t, int, int, int>(
            (arg("thread_pool"), arg("multicast_group"), arg("port"),
             arg("config") = stream_config(),
             arg("interface_address"),
             arg("buffer_size") = T::default_buffer_size,
             arg("ttl") = 1,
             arg("comp_vector") = 0,
             arg("max_poll") = T::default_max_poll))[
            store_handle_postcall<T, thread_pool_handle_wrapper, &thread_pool_handle_wrapper::thread_pool_handle, 1, 2>()])
        .def_readonly("DEFAULT_BUFFER_SIZE", T::default_buffer_size)
        .def_readonly("DEFAULT_MAX_POLL", T::default_max_poll);
}
#endif

template<typename T>
void stream_register(boost::python::class_<T, boost::noncopyable> &stream_class)
{
    using namespace boost::python;
    stream_class.def("set_cnt_sequence", &T::set_cnt_sequence,
                     (arg("next"), arg("step")));
}

template<typename T>
void sync_stream_register(boost::python::class_<T, boost::noncopyable> &stream_class)
{
    using namespace boost::python;
    stream_register(stream_class);
    stream_class.def("send_heap", &T::send_heap, (arg("heap"), arg("cnt") = s_item_pointer_t(-1)));
}

template<typename T>
void async_stream_register(boost::python::class_<T, boost::noncopyable> &stream_class)
{
    using namespace boost::python;
    stream_register(stream_class);
    stream_class
        .add_property("fd", &T::get_fd)
        .def("async_send_heap", &T::async_send_heap,
             (arg("heap"), arg("callback"), arg("cnt") = s_item_pointer_t(-1)))
        .def("flush", &T::flush)
        .def("process_callbacks", &T::process_callbacks);
}

/// Register the send module with Boost.Python
void register_module()
{
    using namespace boost::python;
    using namespace spead2::send;

    // Create the module, and set it as the current boost::python scope so that
    // classes we define are added to this module rather than the root.
    py::object module(py::handle<>(py::borrowed(PyImport_AddModule("spead2._send"))));
    py::scope scope = module;

    class_<heap_wrapper, boost::noncopyable>("Heap", init<flavour>(
            (arg("flavour") = flavour())))
        .add_property("flavour", &heap_wrapper::get_flavour)
        .def("add_item", &heap_wrapper::add_item, arg("item"))
        .def("add_descriptor", &heap_wrapper::add_descriptor,
             (arg("descriptor")))
        .def("add_start", &heap_wrapper::add_start)
        .def("add_end", &heap_wrapper::add_end);

    class_<packet_generator_wrapper, boost::noncopyable>("PacketGenerator", init<heap_wrapper &, item_pointer_t, std::size_t>(
            (arg("heap"), arg("cnt"), arg("max_packet_size")))[
            store_handle_postcall<packet_generator_wrapper, packet_generator_wrapper, &packet_generator_wrapper::heap_handle, 1, 2>()])
        .def("__iter__", objects::identity_function())
        .def(
#if PY_MAJOR_VERSION >= 3
              // Python 3 uses __next__ for the iterator protocol
              "__next__"
#else
              "next"
#endif
              , &packet_generator_wrapper::next);

    class_<stream_config>("StreamConfig", init<
            std::size_t, double, std::size_t, std::size_t>(
                (arg("max_packet_size") = stream_config::default_max_packet_size,
                 arg("rate") = 0.0,
                 arg("burst_size") = stream_config::default_burst_size,
                 arg("max_heaps") = stream_config::default_max_heaps)))
        .add_property("max_packet_size", &stream_config::get_max_packet_size, &stream_config::set_max_packet_size)
        .add_property("rate", &stream_config::get_rate, &stream_config::set_rate)
        .add_property("burst_size", &stream_config::get_burst_size, &stream_config::set_burst_size)
        .add_property("max_heaps", &stream_config::get_max_heaps, &stream_config::set_max_heaps)
        .def_readonly("DEFAULT_MAX_PACKET_SIZE", stream_config::default_max_packet_size)
        .def_readonly("DEFAULT_MAX_HEAPS", stream_config::default_max_heaps)
        .def_readonly("DEFAULT_BURST_SIZE", stream_config::default_burst_size);

    {
        auto stream_class = udp_stream_register<udp_stream_wrapper<stream_wrapper<udp_stream>>>("UdpStream");
        sync_stream_register(stream_class);
    }
    {
        auto stream_class = udp_stream_register<udp_stream_wrapper<asyncio_stream_wrapper<udp_stream>>>("UdpStreamAsyncio");
        async_stream_register(stream_class);
    }

#if SPEAD2_USE_IBV
    {
        auto stream_class = udp_ibv_stream_register<udp_ibv_stream_wrapper<stream_wrapper<udp_ibv_stream>>>("UdpIbvStream");
        sync_stream_register(stream_class);
    }
    {
        auto stream_class = udp_ibv_stream_register<udp_ibv_stream_wrapper<asyncio_stream_wrapper<udp_ibv_stream>>>("UdpIbvStreamAsyncio");
        async_stream_register(stream_class);
    }
#endif

    {
        auto stream_class = class_<bytes_stream, boost::noncopyable>(
            "BytesStream", init<thread_pool_wrapper &, const stream_config &>(
                (arg("thread_pool"), arg("config") = stream_config()))[
            store_handle_postcall<bytes_stream, thread_pool_handle_wrapper, &thread_pool_handle_wrapper::thread_pool_handle, 1, 2>()])
        .def("getvalue", &bytes_stream::getvalue);
        sync_stream_register(stream_class);
    }
}

} // namespace send
} // namespace spead2
