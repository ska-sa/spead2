/* Copyright 2015, 2017, 2019 SKA South Africa
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
#include <boost/optional.hpp>
#include <stdexcept>
#include <mutex>
#include <utility>
#include <memory>
#include <unistd.h>
#include <spead2/send_heap.h>
#include <spead2/send_stream.h>
#include <spead2/send_udp.h>
#include <spead2/send_udp_ibv.h>
#include <spead2/send_tcp.h>
#include <spead2/send_streambuf.h>
#include <spead2/send_inproc.h>
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

static py::object make_io_error(const boost::system::error_code &ec)
{
    if (ec)
    {
        py::object exc_class = py::reinterpret_borrow<py::object>(PyExc_IOError);
        return exc_class(ec.value(), ec.message());
    }
    else
        return py::none();
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

    bool async_send_heap_obj(py::object h, py::object callback, s_item_pointer_t cnt = -1)
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
                callback(make_io_error(item.ec), item.bytes_transferred);
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
    return make_address_no_release(io_service, hostname,
                                   boost::asio::ip::resolver_query_base::flags(0));
}

template<typename Protocol>
static typename Protocol::endpoint make_endpoint(
    boost::asio::io_service &io_service, const std::string &hostname, std::uint16_t port)
{
    return typename Protocol::endpoint(make_address(io_service, hostname), port);
}

template<typename Base>
class udp_stream_wrapper : public Base
{
public:
    udp_stream_wrapper(
        io_service_ref io_service,
        const std::string &hostname,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        const std::string &interface_address)
        : Base(
            std::move(io_service),
            make_endpoint<boost::asio::ip::udp>(*io_service, hostname, port),
            config, buffer_size,
            make_address(*io_service, interface_address))
    {
    }

    udp_stream_wrapper(
        io_service_ref io_service,
        const std::string &hostname,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        const socket_wrapper<boost::asio::ip::udp::socket> &socket)
        : Base(
            std::move(io_service),
            socket.copy(*io_service),
            make_endpoint<boost::asio::ip::udp>(*io_service, hostname, port),
            config, buffer_size)
    {
        deprecation_warning("UdpStream constructor with both buffer_size and socket is deprecated");
    }

    udp_stream_wrapper(
        io_service_ref io_service,
        const std::string &multicast_group,
        std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl)
        : Base(
            std::move(io_service),
            make_endpoint<boost::asio::ip::udp>(*io_service, multicast_group, port),
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
        : Base(
            std::move(io_service),
            make_endpoint<boost::asio::ip::udp>(*io_service, multicast_group, port),
            config, buffer_size, ttl,
            interface_address.empty() ?
                boost::asio::ip::address() :
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
        : Base(
            std::move(io_service),
            make_endpoint<boost::asio::ip::udp>(*io_service, multicast_group, port),
            config, buffer_size, ttl, interface_index)
    {
    }

    udp_stream_wrapper(
        io_service_ref io_service,
        const socket_wrapper<boost::asio::ip::udp::socket> &socket,
        const std::string &hostname,
        std::uint16_t port,
        const stream_config &config)
        : Base(
            std::move(io_service),
            socket.copy(*io_service),
            make_endpoint<boost::asio::ip::udp>(*io_service, hostname, port),
            config)
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
               make_endpoint<boost::asio::ip::udp>(pool->get_io_service(), multicast_group, port),
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
static py::class_<T> udp_stream_register(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    return py::class_<T>(m, name)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, std::uint16_t, const stream_config &, std::size_t, const socket_wrapper<boost::asio::ip::udp::socket> &>(),
             "thread_pool"_a, "hostname"_a, "port"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "socket"_a)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, std::uint16_t, const stream_config &, std::size_t, std::string>(),
             "thread_pool"_a, "hostname"_a, "port"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "interface_address"_a = std::string())
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, std::uint16_t, const stream_config &, std::size_t, int>(),
             "thread_pool"_a, "hostname"_a, "port"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "ttl"_a)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, std::uint16_t, const stream_config &, std::size_t, int, std::string>(),
             "thread_pool"_a, "multicast_group"_a, "port"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "ttl"_a,
             "interface_address"_a)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, std::uint16_t, const stream_config &, std::size_t, int, unsigned int>(),
             "thread_pool"_a, "multicast_group"_a, "port"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "ttl"_a,
             "interface_index"_a)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, const socket_wrapper<boost::asio::ip::udp::socket> &, std::string, std::uint16_t, const stream_config &>(),
             "thread_pool"_a, "socket"_a, "hostname"_a, "port"_a,
             "config"_a = stream_config())
        .def_readonly_static("DEFAULT_BUFFER_SIZE", &T::default_buffer_size);
}

#if SPEAD2_USE_IBV
template<typename T>
static py::class_<T> udp_ibv_stream_register(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    return py::class_<T>(m, name)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::string, std::uint16_t, const stream_config &, std::string, std::size_t, int, int, int>(),
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

template<typename Base>
class tcp_stream_wrapper : public Base
{
public:
    /* All wrapping constructors that use a connect_handler take it as the
     * first argument, to faciliate the meta-programming used by registration
     * code.
     */
    template<typename ConnectHandler>
    tcp_stream_wrapper(
        ConnectHandler&& connect_handler,
        io_service_ref io_service,
        const std::string &hostname, std::uint16_t port,
        const stream_config &config,
        std::size_t buffer_size,
        const std::string &interface_address)
        : Base(std::move(io_service), connect_handler,
               make_endpoint<boost::asio::ip::tcp>(*io_service, hostname, port),
               config, buffer_size, make_address(*io_service, interface_address))
    {
    }

    tcp_stream_wrapper(
        io_service_ref io_service,
        const socket_wrapper<boost::asio::ip::tcp::socket> &socket,
        const stream_config &config)
        : Base(std::move(io_service), socket.copy(*io_service), config)
    {
    }
};

/* This is a different design than the other registration functions, because
 * the TCP sync and async classes are constructed very differently (because of
 * the handling around connecting). The callback is called (several times) with
 * a function object that generates the unique_ptr<T> plus additional arguments
 * to pass to py::class_::def.
 */
template<typename Registrar>
static py::class_<typename Registrar::stream_type> tcp_stream_register(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    typedef typename Registrar::stream_type T;
    py::class_<T> class_(m, name);
    class_
        .def(py::init<std::shared_ptr<thread_pool_wrapper>,
                      const socket_wrapper<boost::asio::ip::tcp::socket> &,
                      const stream_config &>(),
             "thread_pool"_a, "socket"_a, "config"_a = stream_config())
        .def_readonly_static("DEFAULT_BUFFER_SIZE", &T::default_buffer_size);
    Registrar::template apply<
            std::shared_ptr<thread_pool_wrapper>,
            const std::string &, std::uint16_t,
            const stream_config &, std::size_t, const std::string &>(
        class_,
        "thread_pool"_a, "hostname"_a, "port"_a,
        "config"_a = stream_config(),
        "buffer_size"_a = T::default_buffer_size,
        "interface_address"_a = "");
    return class_;
}

// Function object passed to tcp_stream_register to register the synchronous class
class tcp_stream_register_sync
{
private:
    struct connect_state
    {
        semaphore_gil<semaphore> sem;
        boost::system::error_code ec;
    };

public:
    typedef tcp_stream_wrapper<stream_wrapper<tcp_stream>> stream_type;

private:
    /* Template args are explicit, hence no Args&&... */
    template<typename... Args>
    static std::unique_ptr<stream_type> construct(Args... args)
    {
        std::shared_ptr<connect_state> state = std::make_shared<connect_state>();
        auto connect_handler = [state](const boost::system::error_code &ec)
        {
            state->ec = ec;
            state->sem.put();
        };
        std::unique_ptr<stream_type> stream{new stream_type(connect_handler, std::forward<Args>(args)...)};
        state->sem.get();
        if (state->ec)
            throw boost_io_error(state->ec);
        return stream;
    }

public:
    template<typename... Args, typename... Extra>
    static void apply(py::class_<stream_type> &class_, Extra&&... extra)
    {
        class_.def(py::init(&tcp_stream_register_sync::construct<Args...>),
                   std::forward<Extra>(extra)...);
    }
};

// Function object passed to tcp_stream_register to register the asynchronous class
class tcp_stream_register_async
{
private:
    struct connect_state
    {
        py::handle callback;
    };

public:
    typedef tcp_stream_wrapper<asyncio_stream_wrapper<tcp_stream>> stream_type;

private:
    /* Template args are explicit, hence no Args&&... */
    template<typename... Args>
    static std::unique_ptr<stream_type> construct(py::object callback, Args... args)
    {
        std::shared_ptr<connect_state> state = std::make_shared<connect_state>();
        auto connect_handler = [state](boost::system::error_code ec)
        {
            py::gil_scoped_acquire gil;
            py::object callback = py::reinterpret_steal<py::object>(state->callback);
            callback(make_io_error(ec));
        };
        std::unique_ptr<stream_type> stream{
            new stream_type(connect_handler, std::forward<Args>(args)...)};
        /* The state takes over the references. These are dealt with using
         * py::handle rather than py::object to avoid manipulating refcounts
         * without the GIL. Note that while the connect_handler could occur
         * immediately, the GIL serialises access to state.
         */
        state->callback = callback.release();
        return stream;
    }

public:
    template<typename... Args, typename... Extra>
    static void apply(py::class_<stream_type> &class_, Extra&&... extra)
    {
        using namespace pybind11::literals;
        class_.def(py::init(&tcp_stream_register_async::construct<Args...>),
                   "callback"_a, std::forward<Extra>(extra)...);
    }
};

template<typename T>
static py::class_<T> inproc_stream_register(py::module &m, const char *name)
{
    using namespace pybind11::literals;
    return py::class_<T>(m, name)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, std::shared_ptr<inproc_queue>, const stream_config &>(),
             "thread_pool"_a, "queue"_a, "config"_a = stream_config())
        .def_property_readonly("queue", SPEAD2_PTMF(T, get_queue));
}

template<typename T>
static void stream_register(py::class_<T> &stream_class)
{
    using namespace pybind11::literals;
    stream_class.def("set_cnt_sequence", SPEAD2_PTMF(T, set_cnt_sequence),
                     "next"_a, "step"_a);
}

template<typename T>
static void sync_stream_register(py::class_<T> &stream_class)
{
    using namespace pybind11::literals;
    stream_register(stream_class);
    stream_class.def("send_heap", SPEAD2_PTMF(T, send_heap),
                     "heap"_a, "cnt"_a = s_item_pointer_t(-1));
}

template<typename T>
static void async_stream_register(py::class_<T> &stream_class)
{
    using namespace pybind11::literals;
    stream_register(stream_class);
    stream_class
        .def_property_readonly("fd", SPEAD2_PTMF(T, get_fd))
        .def("async_send_heap", SPEAD2_PTMF(T, async_send_heap_obj),
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

    py::class_<heap_wrapper>(m, "Heap")
        .def(py::init<flavour>(), "flavour"_a = flavour())
        .def_property_readonly("flavour", SPEAD2_PTMF(heap_wrapper, get_flavour))
        .def("add_item", SPEAD2_PTMF(heap_wrapper, add_item), "item"_a)
        .def("add_descriptor", SPEAD2_PTMF(heap_wrapper, add_descriptor), "descriptor"_a)
        .def("add_start", SPEAD2_PTMF(heap_wrapper, add_start))
        .def("add_end", SPEAD2_PTMF(heap_wrapper, add_end))
        .def_property("repeat_pointers",
                      SPEAD2_PTMF(heap_wrapper, get_repeat_pointers),
                      SPEAD2_PTMF(heap_wrapper, set_repeat_pointers));

    // keep_alive is safe to use here in spite of pybind/pybind11#856, because
    // the destructor of packet_generator doesn't reference the heap.
    py::class_<packet_generator>(m, "PacketGenerator")
        .def(py::init<heap_wrapper &, item_pointer_t, std::size_t>(),
             "heap"_a, "cnt"_a, "max_packet_size"_a,
             py::keep_alive<1, 2>())
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", &packet_generator_next);

    py::class_<stream_config>(m, "StreamConfig")
        .def(py::init<std::size_t, double, std::size_t, std::size_t, double>(),
             "max_packet_size"_a = stream_config::default_max_packet_size,
             "rate"_a = 0.0,
             "burst_size"_a = stream_config::default_burst_size,
             "max_heaps"_a = stream_config::default_max_heaps,
             "burst_rate_ratio"_a = stream_config::default_burst_rate_ratio)
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
        .def_property("burst_rate_ratio",
                      SPEAD2_PTMF(stream_config, get_burst_rate_ratio),
                      SPEAD2_PTMF(stream_config, set_burst_rate_ratio))
        .def_property_readonly("burst_rate",
                               SPEAD2_PTMF(stream_config, get_burst_rate))
        .def_readonly_static("DEFAULT_MAX_PACKET_SIZE", &stream_config::default_max_packet_size)
        .def_readonly_static("DEFAULT_MAX_HEAPS", &stream_config::default_max_heaps)
        .def_readonly_static("DEFAULT_BURST_SIZE", &stream_config::default_burst_size)
        .def_readonly_static("DEFAULT_BURST_RATE_RATIO", &stream_config::default_burst_rate_ratio);

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
        auto stream_class = tcp_stream_register<tcp_stream_register_sync>(m, "TcpStream");
        sync_stream_register(stream_class);
    }
    {
        auto stream_class = tcp_stream_register<tcp_stream_register_async>(m, "TcpStreamAsyncio");
        async_stream_register(stream_class);
    }

    {
        py::class_<bytes_stream> stream_class(m, "BytesStream");
        stream_class
            .def(py::init<std::shared_ptr<thread_pool_wrapper>, const stream_config &>(),
                 "thread_pool"_a, "config"_a = stream_config())
            .def("getvalue", SPEAD2_PTMF(bytes_stream, getvalue));
        sync_stream_register(stream_class);
    }

    {
        auto stream_class = inproc_stream_register<stream_wrapper<inproc_stream>>(m, "InprocStream");
        sync_stream_register(stream_class);
    }
    {
        auto stream_class = inproc_stream_register<asyncio_stream_wrapper<inproc_stream>>(m, "InprocStreamAsyncio");
        async_stream_register(stream_class);
    }

    return m;
}

} // namespace send
} // namespace spead2
