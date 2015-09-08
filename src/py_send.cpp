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
#include <stdexcept>
#include <mutex>
#include <utility>
#include <memory>
#include "send_heap.h"
#include "send_stream.h"
#include "send_udp.h"
#include "send_streambuf.h"
#include "common_thread_pool.h"
#include "py_common.h"

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
public:
    using Base::Base;

    /// Sends heap synchronously
    void send_heap(const heap_wrapper &h)
    {
        /* A simple future-promise sync here works, but prevents interruption
         * via KeyboardInterrupt. The semaphore needs to be in shared_ptr because
         * if we are interrupted it still needs to exist until the heap is sent.
         */
        auto sent_sem = std::make_shared<semaphore_gil>();
        Base::async_send_heap(h, [sent_sem] (const boost::system::error_code &, item_pointer_t)
        {
            sent_sem->put();
        });
        while (sent_sem->get() == -1)
        {
            // retry if interrupted for other reason
        }
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

    semaphore_gil sem;
    std::vector<callback_item> callbacks;
    std::mutex callbacks_mutex;

    // Prevent copying: the callbacks vector cannot sanely be copied
    asyncio_stream_wrapper(const asyncio_stream_wrapper &) = delete;
    asyncio_stream_wrapper &operator=(const asyncio_stream_wrapper &) = delete;
public:
    using Base::Base;

    int get_fd() const { return sem.get_fd(); }

    void async_send_heap(py::object h, py::object callback)
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
        Base::async_send_heap(h2(), [this, callback_ptr, h_ptr] (
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
        });
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

template<typename Base>
class udp_stream_wrapper : public Base
{
private:
    boost::python::handle<> thread_pool_handle;
    friend void register_module();

    static boost::asio::ip::udp::endpoint make_endpoint(
        boost::asio::io_service &io_service, const std::string &hostname, int port);

public:
    udp_stream_wrapper(
        thread_pool &pool,
        const std::string &hostname,
        int port,
        const stream_config &config = stream_config(),
        std::size_t buffer_size = Base::default_buffer_size)
        : Base(
            pool.get_io_service(),
            make_endpoint(pool.get_io_service(), hostname, port),
            config, buffer_size)
    {
    }
};

template<typename Base>
boost::asio::ip::udp::endpoint udp_stream_wrapper<Base>::make_endpoint(
    boost::asio::io_service &io_service, const std::string &hostname, int port)
{
    using boost::asio::ip::udp;
    udp::endpoint endpoint(boost::asio::ip::address_v4::any(), port);
    udp::resolver resolver(io_service);
    udp::resolver::query query(hostname, "", udp::resolver::query::address_configured);
    endpoint.address(resolver.resolve(query)->endpoint().address());
    return endpoint;
}

class bytes_stream : private std::stringbuf, public stream_wrapper<streambuf_stream>
{
private:
    boost::python::handle<> thread_pool_handle;
    friend void register_module();

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
        .def("add_end", &heap_wrapper::add_end);

    class_<packet_generator_wrapper, boost::noncopyable>("PacketGenerator", init<heap_wrapper &, item_pointer_t, std::size_t>(
            (arg("heap"), arg("cnt"), arg("max_packet_size")))[
            store_handle_postcall<packet_generator_wrapper, &packet_generator_wrapper::heap_handle, 1, 2>()])
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
        typedef udp_stream_wrapper<stream_wrapper<udp_stream> > T;
        class_<T, boost::noncopyable>("UdpStream", init<
                thread_pool_wrapper &, std::string, int, const stream_config &, std::size_t>(
                    (arg("thread_pool"), arg("hostname"), arg("port"),
                     arg("config") = stream_config(),
                     arg("buffer_size") = T::default_buffer_size))[
                store_handle_postcall<T, &T::thread_pool_handle, 1, 2>()])
            .def("send_heap", &T::send_heap, arg("heap"))
            .def_readonly("DEFAULT_BUFFER_SIZE", T::default_buffer_size);
    }

    {
        typedef udp_stream_wrapper<asyncio_stream_wrapper<udp_stream> > T;
        class_<T, boost::noncopyable>("UdpStreamAsyncio", init<
                thread_pool_wrapper &, std::string, int, const stream_config &, std::size_t>(
                    (arg("thread_pool"), arg("hostname"), arg("port"),
                     arg("config") = stream_config(),
                     arg("buffer_size") = T::default_buffer_size))[
                store_handle_postcall<T, &T::thread_pool_handle, 1, 2>()])
            .add_property("fd", &T::get_fd)
            .def("async_send_heap", &T::async_send_heap, arg("heap"))
            .def("flush", &T::flush)
            .def("process_callbacks", &T::process_callbacks)
            .def_readonly("DEFAULT_BUFFER_SIZE", T::default_buffer_size);
    }

    class_<bytes_stream, boost::noncopyable>("BytesStream", init<
                thread_pool_wrapper &, const stream_config &>(
                    (arg("thread_pool"), arg("config") = stream_config()))[
                store_handle_postcall<bytes_stream, &bytes_stream::thread_pool_handle, 1, 2>()])
        .def("getvalue", &bytes_stream::getvalue)
        .def("send_heap", &bytes_stream::send_heap, arg("heap"));
}

} // namespace send
} // namespace spead2
