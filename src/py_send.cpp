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

namespace py = pybind11;

namespace spead2
{
namespace send
{

class heap_wrapper : public heap
{
private:
    std::vector<py::buffer_info> item_buffers;

public:
    using heap::heap;
    void add_item(py::object item);
    void add_descriptor(py::object descriptor);
    flavour get_flavour() const;
};

void heap_wrapper::add_item(py::object item)
{
    std::int64_t id = item.attr("id").cast<std::int64_t>();
    py::buffer buffer = item.attr("to_buffer")().cast<py::buffer>();
    bool allow_immediate = item.attr("allow_immediate")().cast<bool>();
    item_buffers.emplace_back(request_buffer_info(buffer, PyBUF_C_CONTIGUOUS));
    heap::add_item(id, item_buffers.back().ptr,
                   item_buffers.back().itemsize * item_buffers.back().size,
                   allow_immediate);
}

void heap_wrapper::add_descriptor(py::object object)
{
    heap::add_descriptor(object.attr("to_raw")(heap::get_flavour()).cast<descriptor>());
}

flavour heap_wrapper::get_flavour() const
{
    return heap::get_flavour();
}

py::bytes packet_generator_next(packet_generator &gen)
{
    packet pkt = gen.next_packet();
    if (pkt.buffers.empty())
        throw py::stop_iteration();
    return py::bytes(std::string(boost::asio::buffers_begin(pkt.buffers),
                                 boost::asio::buffers_end(pkt.buffers)));
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
        py::handle callback;
        py::handle h;  // heap: kept here because it can only be freed with the GIL
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
        /* Normally the callback should not refer to this, since it could have
         * been reaped by the time the callback occurs. We rely on Python to
         * hang on to a reference to self.
         *
         * The callback and heap are passed around by raw reference, because
         * it is not safe to use incref/decref operations without the GIL, and
         * attempting to use py::object instead of py::handle tends to cause
         * these operations to occur without it being obvious.
         */
        py::handle h_ptr = h.ptr();
        py::handle callback_ptr = callback.ptr();
        h_ptr.inc_ref();
        callback_ptr.inc_ref();
        return Base::async_send_heap(h.cast<heap_wrapper &>(), [this, callback_ptr, h_ptr] (
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
                item.h.dec_ref();
                item.h = py::handle();
                py::object callback = py::reinterpret_steal<py::object>(item.callback);
                item.callback = py::handle();
                py::object exc;
                if (item.ec)
                {
                    py::object exc_class = py::reinterpret_borrow<py::object>(PyExc_IOError);
                    exc = exc_class(item.ec.value(), item.ec.message());
                }
                else
                    exc = py::none();
                callback(exc, item.bytes_transferred);
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
                if (item.h)
                    item.h.dec_ref();
                if (item.callback)
                    item.callback.dec_ref();
            }
            throw;
        }
    }

    ~asyncio_stream_wrapper()
    {
        for (const callback_item &item : callbacks)
        {
            item.h.dec_ref();
            item.callback.dec_ref();
        }
    }
};

static boost::asio::ip::address make_address(
    boost::asio::io_service &io_service, const std::string &hostname)
{
    py::gil_scoped_release gil;

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
        int fd = socket.attr("fileno")().cast<int>();
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
class udp_stream_wrapper : public Base
{
private:
    /* Intermediate chained constructors that have the hostname and port
     * converted to an endpoint, so that it can be used in turn to
     * construct the asio socket.
     */
    udp_stream_wrapper(
        io_service_ref io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        std::size_t buffer_size,
        py::object socket)
        : Base(
            std::move(io_service),
            make_socket(*io_service, endpoint.protocol(), std::move(socket)),
            endpoint,
            config, buffer_size)
    {
    }

    udp_stream_wrapper(
        io_service_ref io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl)
        : Base(std::move(io_service), endpoint, config, buffer_size, ttl)
    {
    }

    template<typename T>
    udp_stream_wrapper(
        io_service_ref io_service,
        const boost::asio::ip::udp::endpoint &endpoint,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        const T &interface)
        : Base(std::move(io_service), endpoint, config, buffer_size, ttl, interface)
    {
    }

public:
    udp_stream_wrapper(
        io_service_ref io_service,
        const std::string &hostname,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        py::object socket)
        : udp_stream_wrapper(
            std::move(io_service), make_endpoint(*io_service, hostname, port),
            config, buffer_size, std::move(socket))
    {
    }

    udp_stream_wrapper(
        io_service_ref io_service,
        const std::string &multicast_group,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl)
        : udp_stream_wrapper(
            std::move(io_service), make_endpoint(*io_service, multicast_group, port),
            config, buffer_size, ttl)
    {
    }

    udp_stream_wrapper(
        io_service_ref io_service,
        const std::string &multicast_group,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        const std::string &interface_address)
        : udp_stream_wrapper(
            std::move(io_service), make_endpoint(*io_service, multicast_group, port),
            config, buffer_size, ttl,
            interface_address.empty() ?
                boost::asio::ip::address_v4::any() :
                make_address(*io_service, interface_address))
    {
    }

    udp_stream_wrapper(
        io_service_ref io_service,
        const std::string &multicast_group,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        unsigned int interface_index)
        : udp_stream_wrapper(
            std::move(io_service), make_endpoint(*io_service, multicast_group, port),
            config, buffer_size, ttl, interface_index)
    {
    }
};

#if SPEAD2_USE_IBV
template<typename Base>
class udp_ibv_stream_wrapper : public Base
{
public:
    udp_ibv_stream_wrapper(
        std::shared_ptr<thread_pool> pool,
        const std::string &multicast_group,
        std::uint16_t port,
        const stream_config &config,
        const std::string &interface_address,
        std::size_t buffer_size,
        int ttl,
        int comp_vector,
        int max_poll)
        : Base(std::move(pool),
               make_endpoint(pool->get_io_service(), multicast_group, port),
               config,
               make_address(pool->get_io_service(), interface_address),
               buffer_size, ttl, comp_vector, max_poll)
    {
    }
};
#endif

class bytes_stream : private std::stringbuf, public stream_wrapper<streambuf_stream>
{
public:
    bytes_stream(std::shared_ptr<thread_pool> pool, const stream_config &config = stream_config())
        : stream_wrapper<streambuf_stream>(std::move(pool), *this, config)
    {
    }

    py::bytes getvalue() const
    {
        return str();
    }
};

template<typename T>
static spead2::class_<T> udp_stream_register(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    return spead2::class_<T>(m, name)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, int, const stream_config &, std::size_t, py::object>(),
             "thread_pool"_a, "hostname"_a, "port"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "socket"_a = py::none())
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, int, const stream_config &, std::size_t, int>(),
             "thread_pool"_a, "hostname"_a, "port"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "ttl"_a)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, int, const stream_config &, std::size_t, int, std::string>(),
             "thread_pool"_a, "multicast_group"_a, "port"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "ttl"_a,
             "interface_address"_a)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, int, const stream_config &, std::size_t, int, unsigned int>(),
             "thread_pool"_a, "multicast_group"_a, "port"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "ttl"_a,
             "interface_index"_a)
        .def_readonly_static("DEFAULT_BUFFER_SIZE", &T::default_buffer_size);
}

#if SPEAD2_USE_IBV
template<typename T>
static spead2::class_<T> udp_ibv_stream_register(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    return spead2::class_<T>(m, name)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, int, const stream_config &, std::string, std::size_t, int, int, int>(),
             "thread_pool"_a, "multicast_group"_a, "port"_a,
             "config"_a = stream_config(),
             "interface_address"_a,
             "buffer_size"_a = T::default_buffer_size,
             "ttl"_a = 1,
             "comp_vector"_a = 0,
             "max_poll"_a = T::default_max_poll)
        .def_readonly_static("DEFAULT_BUFFER_SIZE", &T::default_buffer_size)
        .def_readonly_static("DEFAULT_MAX_POLL", &T::default_max_poll);
}
#endif

template<typename T>
static void stream_register(spead2::class_<T> &stream_class)
{
    using namespace pybind11::literals;
    stream_class.def("set_cnt_sequence", SPEAD2_PTMF(T, set_cnt_sequence),
                     "next"_a, "step"_a);
}

template<typename T>
static void sync_stream_register(spead2::class_<T> &stream_class)
{
    using namespace pybind11::literals;
    stream_register(stream_class);
    stream_class.def("send_heap", SPEAD2_PTMF(T, send_heap),
                     "heap"_a, "cnt"_a = s_item_pointer_t(-1));
}

template<typename T>
static void async_stream_register(spead2::class_<T> &stream_class)
{
    using namespace pybind11::literals;
    stream_register(stream_class);
    stream_class
        .def_property_readonly("fd", SPEAD2_PTMF(T, get_fd))
        .def("async_send_heap", SPEAD2_PTMF(T, async_send_heap),
             "heap"_a, "callback"_a, "cnt"_a = s_item_pointer_t(-1))
        .def("flush", SPEAD2_PTMF(T, flush))
        .def("process_callbacks", SPEAD2_PTMF(T, process_callbacks));
}

/// Register the send module with Boost.Python
py::module register_module(py::module &parent)
{
    using namespace pybind11::literals;
    using namespace spead2::send;

    py::module m = parent.def_submodule("send");

    spead2::class_<heap_wrapper>(m, "Heap")
        .def(py::init<flavour>(), "flavour"_a = flavour())
        .def_property_readonly("flavour", &heap_wrapper::get_flavour)
        .def("add_item", SPEAD2_PTMF(heap_wrapper, add_item), "item"_a)
        .def("add_descriptor", SPEAD2_PTMF(heap_wrapper, add_descriptor), "descriptor"_a)
        .def("add_start", SPEAD2_PTMF(heap_wrapper, add_start))
        .def("add_end", SPEAD2_PTMF(heap_wrapper, add_end));

    // keep_alive is safe to use here in spite of pybind/pybind11#856, because
    // the destructor of packet_generator doesn't reference the heap.
    spead2::class_<packet_generator>(m, "PacketGenerator")
        .def(py::init<heap_wrapper &, item_pointer_t, std::size_t>(),
             "heap"_a, "cnt"_a, "max_packet_size"_a,
             py::keep_alive<1, 2>())
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", &packet_generator_next);

    spead2::class_<stream_config>(m, "StreamConfig")
        .def(py::init<std::size_t, double, std::size_t, std::size_t>(),
             "max_packet_size"_a = stream_config::default_max_packet_size,
             "rate"_a = 0.0,
             "burst_size"_a = stream_config::default_burst_size,
             "max_heaps"_a = stream_config::default_max_heaps)
        .def_property("max_packet_size",
                      SPEAD2_PTMF(stream_config, get_max_packet_size),
                      SPEAD2_PTMF(stream_config, set_max_packet_size))
        .def_property("rate",
                      SPEAD2_PTMF(stream_config, get_rate),
                      SPEAD2_PTMF(stream_config, set_rate))
        .def_property("burst_size",
                      SPEAD2_PTMF(stream_config, get_burst_size),
                      SPEAD2_PTMF(stream_config, set_burst_size))
        .def_property("max_heaps",
                      SPEAD2_PTMF(stream_config, get_max_heaps),
                      SPEAD2_PTMF(stream_config, set_max_heaps))
        .def_readonly_static("DEFAULT_MAX_PACKET_SIZE", &stream_config::default_max_packet_size)
        .def_readonly_static("DEFAULT_MAX_HEAPS", &stream_config::default_max_heaps)
        .def_readonly_static("DEFAULT_BURST_SIZE", &stream_config::default_burst_size);

    {
        auto stream_class = udp_stream_register<udp_stream_wrapper<stream_wrapper<udp_stream>>>(m, "UdpStream");
        sync_stream_register(stream_class);
    }
    {
        auto stream_class = udp_stream_register<udp_stream_wrapper<asyncio_stream_wrapper<udp_stream>>>(m, "UdpStreamAsyncio");
        async_stream_register(stream_class);
    }

#if SPEAD2_USE_IBV
    {
        auto stream_class = udp_ibv_stream_register<udp_ibv_stream_wrapper<stream_wrapper<udp_ibv_stream>>>(m, "UdpIbvStream");
        sync_stream_register(stream_class);
    }
    {
        auto stream_class = udp_ibv_stream_register<udp_ibv_stream_wrapper<asyncio_stream_wrapper<udp_ibv_stream>>>(m, "UdpIbvStreamAsyncio");
        async_stream_register(stream_class);
    }
#endif

    {
        spead2::class_<bytes_stream> stream_class(m, "BytesStream");
        stream_class
            .def(py::init<std::shared_ptr<thread_pool_wrapper>, const stream_config &>(),
                 "thread_pool"_a, "config"_a = stream_config())
            .def("getvalue", SPEAD2_PTMF(bytes_stream, getvalue));
        sync_stream_register(stream_class);
    }

    return m;
}

} // namespace send
} // namespace spead2
