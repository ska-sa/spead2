/* Copyright 2015, 2017, 2019-2021, 2023-2025 National Research Foundation (SARAO)
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
#include <vector>
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
#include "common_unique.h"

namespace py = pybind11;

namespace spead2
{
namespace send
{

using spead2::detail::discard_result;

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
    auto scratch = spead2::detail::make_unique_for_overwrite<std::uint8_t[]>(gen.get_max_packet_size());
    std::vector<boost::asio::const_buffer> buffers;
    gen.next_packet(scratch.get(), buffers);
    if (buffers.empty())
        throw py::stop_iteration();
    return py::bytes(std::string(boost::asio::buffers_begin(buffers),
                                 boost::asio::buffers_end(buffers)));
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

class heap_reference_list
{
private:
    std::vector<heap_reference> heaps;
    // Python references to the heaps, to keep them alive
    std::vector<py::object> objects;

    heap_reference_list(std::vector<heap_reference> heaps, std::vector<py::object> objects)
        : heaps(std::move(heaps)), objects(std::move(objects)) {}
public:
    heap_reference_list(std::vector<heap_reference> heaps);
    const std::vector<heap_reference> &get_heaps() const { return heaps; }
    std::size_t size() const { return heaps.size(); }
    heap_reference_list get_slice(const py::slice &slice) const;
};

heap_reference_list::heap_reference_list(std::vector<heap_reference> heaps)
{
    objects.reserve(heaps.size());
    for (const heap_reference &h : heaps)
        objects.push_back(py::cast(static_cast<const heap_wrapper *>(&h.heap)));
    this->heaps = std::move(heaps);
}

heap_reference_list heap_reference_list::get_slice(const py::slice &slice) const
{
    std::size_t start, stop, step, slicelength;
    if (!slice.compute(heaps.size(), &start, &stop, &step, &slicelength))
        throw py::error_already_set();
    std::vector<heap_reference> new_heaps;
    std::vector<py::object> new_objects;
    new_heaps.reserve(slicelength);
    new_objects.reserve(slicelength);
    for (std::size_t i = 0; i < slicelength; i++)
    {
        new_heaps.push_back(heaps[start]);
        new_objects.push_back(objects[start]);
        start += step;
    }
    return heap_reference_list(std::move(new_heaps), std::move(new_objects));
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
        semaphore sem;
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
    item_pointer_t send_heap(
        const heap_wrapper &h,
        s_item_pointer_t cnt = -1,
        std::size_t substream_index = 0,
        double rate = -1.0)
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
        }, cnt, substream_index, rate);
        semaphore_get(state->sem, gil_release_tag());
        if (state->ec)
            throw boost_io_error(state->ec);
        else
            return state->bytes_transferred;
    }

    /// Sends multiple heaps synchronously
    item_pointer_t send_heaps(const std::vector<heap_reference> &heaps, group_mode mode)
    {
        // See comments in send_heap
        auto state = std::make_shared<callback_state>();
        Base::async_send_heaps(
            heaps.begin(), heaps.end(),
            [state] (const boost::system::error_code &ec, item_pointer_t bytes_transferred)
            {
                state->ec = ec;
                state->bytes_transferred = bytes_transferred;
                state->sem.put();
            }, mode);
        semaphore_get(state->sem, gil_release_tag());
        if (state->ec)
            throw boost_io_error(state->ec);
        else
            return state->bytes_transferred;
    }

    /// Sends multiple heaps synchronously, from a pre-built heap_reference_list
    item_pointer_t send_heaps_hrl(const heap_reference_list &heaps, group_mode mode)
    {
        return send_heaps(heaps.get_heaps(), mode);
    }
};

struct callback_item
{
    py::handle callback;
    std::vector<py::handle> heaps;  // kept here because they can only be freed with the GIL
    boost::system::error_code ec;
    item_pointer_t bytes_transferred;
};

static void free_callback_items(const std::vector<callback_item> &callbacks)
{
    for (const callback_item &item : callbacks)
    {
        for (py::handle h : item.heaps)
            h.dec_ref();
        if (item.callback)
            item.callback.dec_ref();
    }
}

template<typename Base>
class asyncio_stream_wrapper : public Base
{
private:
    semaphore_fd sem;
    std::vector<callback_item> callbacks;
    std::mutex callbacks_mutex;

    // Prevent copying: the callbacks vector cannot sanely be copied
    asyncio_stream_wrapper(const asyncio_stream_wrapper &) = delete;
    asyncio_stream_wrapper &operator=(const asyncio_stream_wrapper &) = delete;

    void handler(py::handle callback_ptr, std::vector<py::handle> h_ptr,
                 const boost::system::error_code &ec, item_pointer_t bytes_transferred)
    {
        bool was_empty;
        {
            std::unique_lock<std::mutex> lock(callbacks_mutex);
            was_empty = callbacks.empty();
            callbacks.push_back(callback_item{callback_ptr, std::move(h_ptr), ec, bytes_transferred});
        }
        if (was_empty)
            sem.put();
    }

public:
    using Base::Base;

    int get_fd() const { return sem.get_fd(); }

    bool async_send_heap_obj(
        py::object h,
        py::object callback,
        s_item_pointer_t cnt = -1,
        std::size_t substream_index = 0,
        double rate = -1.0)
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
        return Base::async_send_heap(
            h.cast<const heap_wrapper &>(),
            [this, callback_ptr, h_ptr] (const boost::system::error_code &ec, item_pointer_t bytes_transferred)
            {
                handler(callback_ptr, {h_ptr}, ec, bytes_transferred);
            },
            cnt, substream_index, rate);
    }

    bool async_send_heaps_obj(const std::vector<heap_reference> &heaps,
                              py::object callback, group_mode mode)
    {
        // See comments in async_send_heap_obj
        std::vector<py::handle> h_ptrs;
        h_ptrs.reserve(heaps.size());
        for (const auto &h : heaps)
            h_ptrs.push_back(py::cast(static_cast<const heap_wrapper *>(&h.heap)).release());
        py::handle callback_ptr = callback.ptr();
        callback_ptr.inc_ref();
        return Base::async_send_heaps(
            heaps.begin(), heaps.end(),
            [this, callback_ptr, h_ptrs = std::move(h_ptrs)] (const boost::system::error_code &ec, item_pointer_t bytes_transferred)
            {
                handler(callback_ptr, std::move(h_ptrs), ec, bytes_transferred);
            },
            mode);
    }

    // Overload that takes a HeapReferenceList
    bool async_send_heaps_hrl(const heap_reference_list &heaps,
                              py::object callback, group_mode mode)
    {
        /* In this overload, we just keep the heap_reference_list alive (in Python),
         * and it in turn keeps the individual heaps alive - this requires less
         * reference counting.
         */
        py::handle h_ptr = py::cast(&heaps).release();
        py::handle callback_ptr = callback.ptr();
        callback_ptr.inc_ref();
        return Base::async_send_heaps(
            heaps.get_heaps().begin(), heaps.get_heaps().end(),
            [this, callback_ptr, h_ptr] (const boost::system::error_code &ec, item_pointer_t bytes_transferred)
            {
                handler(callback_ptr, {h_ptr}, ec, bytes_transferred);
            },
            mode);
    }

    void process_callbacks()
    {
        semaphore_get(sem, gil_release_tag());
        std::vector<callback_item> current_callbacks;
        {
            std::unique_lock<std::mutex> lock(callbacks_mutex);
            current_callbacks.swap(callbacks);
        }
        try
        {
            for (callback_item &item : current_callbacks)
            {
                while (!item.heaps.empty())
                {
                    item.heaps.back().dec_ref();
                    item.heaps.pop_back();
                }
                item.heaps.shrink_to_fit();
                py::object callback = py::reinterpret_steal<py::object>(item.callback);
                item.callback = py::handle();
                callback(make_io_error(item.ec), item.bytes_transferred);
            }
        }
        catch (py::error_already_set &e)
        {
            log_warning("send callback raised Python exception; expect deadlocks!");
            free_callback_items(current_callbacks);
            throw;
        }
        catch (std::bad_alloc &e)
        {
            /* If we're out of memory we might not be able to construct a log
             * message. Just rely on Python to report an error.
             */
            free_callback_items(current_callbacks);
            throw;
        }
        catch (std::exception &e)
        {
            log_warning("unexpected error in process_callbacks: %1%", e.what());
            free_callback_items(current_callbacks);
            throw;
        }
    }

    ~asyncio_stream_wrapper()
    {
        for (const callback_item &item : callbacks)
        {
            for (py::handle h : item.heaps)
                h.dec_ref();
            item.callback.dec_ref();
        }
    }
};

static boost::asio::ip::address make_address(
    boost::asio::io_context &io_context, const std::string &hostname)
{
    py::gil_scoped_release gil;
    return make_address_no_release(io_context, hostname,
                                   boost::asio::ip::resolver_query_base::flags(0));
}

template<typename Protocol>
static typename Protocol::endpoint make_endpoint(
    boost::asio::io_context &io_context, const std::string &hostname, std::uint16_t port)
{
    return typename Protocol::endpoint(make_address(io_context, hostname), port);
}

template<typename Protocol>
static std::vector<typename Protocol::endpoint> make_endpoints(
    boost::asio::io_context &io_context, const std::vector<std::pair<std::string, std::uint16_t>> &endpoints)
{
    std::vector<typename Protocol::endpoint> out;
    out.reserve(endpoints.size());
    for (const auto &[host, port] : endpoints)
        out.push_back(make_endpoint<Protocol>(io_context, host, port));
    return out;
}

template<typename Base>
class udp_stream_wrapper : public Base
{
public:
    udp_stream_wrapper(
        io_context_ref io_context,
        const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
        const stream_config &config,
        std::size_t buffer_size,
        const std::string &interface_address)
        : Base(
            io_context,
            make_endpoints<boost::asio::ip::udp>(*io_context, endpoints),
            config, buffer_size,
            make_address(*io_context, interface_address))
    {
    }

    udp_stream_wrapper(
        io_context_ref io_context,
        const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl)
        : Base(
            io_context,
            make_endpoints<boost::asio::ip::udp>(*io_context, endpoints),
            config, buffer_size, ttl)
    {
    }

    udp_stream_wrapper(
        io_context_ref io_context,
        const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        const std::string &interface_address)
        : Base(
            io_context,
            make_endpoints<boost::asio::ip::udp>(*io_context, endpoints),
            config, buffer_size, ttl,
            interface_address.empty() ?
                boost::asio::ip::address() :
                make_address(*io_context, interface_address))
    {
    }

    udp_stream_wrapper(
        io_context_ref io_context,
        const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
        const stream_config &config,
        std::size_t buffer_size,
        int ttl,
        unsigned int interface_index)
        : Base(
            io_context,
            make_endpoints<boost::asio::ip::udp>(*io_context, endpoints),
            config, buffer_size, ttl, interface_index)
    {
    }

    udp_stream_wrapper(
        io_context_ref io_context,
        const socket_wrapper<boost::asio::ip::udp::socket> &socket,
        const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
        const stream_config &config)
        : Base(
            io_context,
            socket.copy(*io_context),
            make_endpoints<boost::asio::ip::udp>(*io_context, endpoints),
            config)
    {
    }
};

#if SPEAD2_USE_IBV

/* Managing the endpoint and memory region lists requires some sleight of
 * hand. We store a separate copy in the wrapper in a Python-centric format.
 * When constructing the stream, we make a copy with the C++ view.
 */
class udp_ibv_config_wrapper : public udp_ibv_config
{
public:
    std::vector<std::pair<std::string, std::uint16_t>> py_endpoints;
    std::vector<py::buffer> py_memory_regions;
    std::string py_interface_address;
};

template<typename Base>
class udp_ibv_stream_wrapper : public Base
{
private:
    // Keeps the buffer requests alive
    std::vector<py::buffer_info> buffer_infos;

public:
    udp_ibv_stream_wrapper(
        std::shared_ptr<thread_pool> pool,
        const stream_config &config,
        const udp_ibv_config &ibv_config,
        std::vector<py::buffer_info> &&buffer_infos)
        : Base(pool,
               config,
               ibv_config),
        buffer_infos(std::move(buffer_infos))
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
static py::class_<T, stream> udp_stream_register(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    return py::class_<T, stream>(m, name)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, const std::vector<std::pair<std::string, std::uint16_t>> &, const stream_config &, std::size_t, std::string>(),
             "thread_pool"_a.none(false), "endpoints"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "interface_address"_a = std::string())
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, const std::vector<std::pair<std::string, std::uint16_t>> &, const stream_config &, std::size_t, int>(),
             "thread_pool"_a.none(false), "endpoints"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "ttl"_a)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, const std::vector<std::pair<std::string, std::uint16_t>> &, const stream_config &, std::size_t, int, std::string>(),
             "thread_pool"_a.none(false), "endpoints"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "ttl"_a,
             "interface_address"_a)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, const std::vector<std::pair<std::string, std::uint16_t>> &, const stream_config &, std::size_t, int, unsigned int>(),
             "thread_pool"_a.none(false), "endpoints"_a,
             "config"_a = stream_config(),
             "buffer_size"_a = T::default_buffer_size,
             "ttl"_a,
             "interface_index"_a)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, const socket_wrapper<boost::asio::ip::udp::socket> &, const std::vector<std::pair<std::string, std::uint16_t>> &, const stream_config &>(),
             "thread_pool"_a.none(false), "socket"_a, "endpoints"_a,
             "config"_a = stream_config())

        .def_readonly_static("DEFAULT_BUFFER_SIZE", &T::default_buffer_size);
}

#if SPEAD2_USE_IBV
template<typename T>
static py::class_<T, stream> udp_ibv_stream_register(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    return py::class_<T, stream>(m, name)
        .def(py::init([](std::shared_ptr<thread_pool_wrapper> thread_pool,
                         const stream_config &config,
                         const udp_ibv_config_wrapper &ibv_config_wrapper)
            {
                udp_ibv_config ibv_config = ibv_config_wrapper;
                ibv_config.set_endpoints(
                    make_endpoints<boost::asio::ip::udp>(
                        thread_pool->get_io_context(),
                        ibv_config_wrapper.py_endpoints));
                ibv_config.set_interface_address(
                    make_address(thread_pool->get_io_context(),
                                 ibv_config_wrapper.py_interface_address));
                std::vector<std::pair<const void *, std::size_t>> regions;
                std::vector<py::buffer_info> buffer_infos;
                regions.reserve(ibv_config_wrapper.py_memory_regions.size());
                buffer_infos.reserve(regions.size());
                for (auto &buffer : ibv_config_wrapper.py_memory_regions)
                {
                    buffer_infos.push_back(request_buffer_info(buffer, PyBUF_C_CONTIGUOUS));
                    regions.emplace_back(
                        buffer_infos.back().ptr,
                        buffer_infos.back().itemsize * buffer_infos.back().size);
                }
                ibv_config.set_memory_regions(regions);

                return new T(std::move(thread_pool), config, ibv_config, std::move(buffer_infos));
            }),
            "thread_pool"_a.none(false),
            "config"_a = stream_config(),
            "udp_ibv_config"_a);
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
        io_context_ref io_context,
        const std::vector<std::pair<std::string, std::uint16_t>> &endpoints,
        const stream_config &config,
        std::size_t buffer_size,
        const std::string &interface_address)
        : Base(io_context, std::forward<ConnectHandler>(connect_handler),
               make_endpoints<boost::asio::ip::tcp>(*io_context, endpoints),
               config, buffer_size, make_address(*io_context, interface_address))
    {
    }

    tcp_stream_wrapper(
        io_context_ref io_context,
        const socket_wrapper<boost::asio::ip::tcp::socket> &socket,
        const stream_config &config)
        : Base(io_context, socket.copy(*io_context), config)
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
static py::class_<typename Registrar::stream_type, stream> tcp_stream_register(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    typedef typename Registrar::stream_type T;
    py::class_<T, stream> class_(m, name);
    class_
        .def(py::init<std::shared_ptr<thread_pool_wrapper>,
                      const socket_wrapper<boost::asio::ip::tcp::socket> &,
                      const stream_config &>(),
             "thread_pool"_a.none(false), "socket"_a, "config"_a = stream_config())
        .def_readonly_static("DEFAULT_BUFFER_SIZE", &T::default_buffer_size);
    Registrar::template apply<
            std::shared_ptr<thread_pool_wrapper>,
            const std::vector<std::pair<std::string, std::uint16_t>> &,
            const stream_config &, std::size_t, const std::string &>(
        class_,
        "thread_pool"_a.none(false), "endpoints"_a,
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
        semaphore sem;
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
        auto stream = std::make_unique<stream_type>(connect_handler, std::forward<Args>(args)...);
        semaphore_get(state->sem, gil_release_tag());
        if (state->ec)
            throw boost_io_error(state->ec);
        return stream;
    }

public:
    template<typename... Args, typename... Extra>
    static void apply(py::class_<stream_type, stream> &class_, Extra&&... extra)
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
        auto stream = std::make_unique<stream_type>(connect_handler, std::forward<Args>(args)...);
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
    static void apply(py::class_<stream_type, stream> &class_, Extra&&... extra)
    {
        using namespace pybind11::literals;
        class_.def(py::init(&tcp_stream_register_async::construct<Args...>),
                   "callback"_a, std::forward<Extra>(extra)...);
    }
};

template<typename T>
static py::class_<T, stream> inproc_stream_register(py::module &m, const char *name)
{
    using namespace pybind11::literals;
    return py::class_<T, stream>(m, name)
        .def(py::init<std::shared_ptr<thread_pool_wrapper>, const std::vector<std::shared_ptr<inproc_queue>> &, const stream_config &>(),
             "thread_pool"_a.none(false), "queues"_a, "config"_a = stream_config())
        .def_property_readonly("queues", &T::get_queues);
}

template<typename T>
static void sync_stream_register(py::class_<T, stream> &stream_class)
{
    using namespace pybind11::literals;
    stream_class.def("send_heap", &T::send_heap,
                     "heap"_a, "cnt"_a = s_item_pointer_t(-1),
                     "substream_index"_a = std::size_t(0),
                     "rate"_a = -1.0);
    stream_class.def("send_heaps", &T::send_heaps_hrl,
                     "heaps"_a, "mode"_a);
    stream_class.def("send_heaps", &T::send_heaps,
                     "heaps"_a, "mode"_a);
}

template<typename T>
static void async_stream_register(py::class_<T, stream> &stream_class)
{
    using namespace pybind11::literals;
    stream_class
        .def_property_readonly("fd", &T::get_fd)
        .def("async_send_heap", &T::async_send_heap_obj,
             "heap"_a, "callback"_a, "cnt"_a = s_item_pointer_t(-1),
             "substream_index"_a = std::size_t(0),
             "rate"_a = -1.0)
        .def("async_send_heaps", &T::async_send_heaps_hrl,
             "heaps"_a, "callback"_a, "mode"_a)
        .def("async_send_heaps", &T::async_send_heaps_obj,
             "heaps"_a, "callback"_a, "mode"_a)
        .def("flush", &T::flush)
        .def("process_callbacks", &T::process_callbacks);
}

/// Register the send module with Boost.Python
py::module register_module(py::module &parent)
{
    using namespace pybind11::literals;

    py::module m = parent.def_submodule("send");

    py::class_<heap_wrapper>(m, "Heap")
        .def(py::init<flavour>(), "flavour"_a = flavour())
        .def_property_readonly("flavour", &heap_wrapper::get_flavour)
        .def("add_item", &heap_wrapper::add_item, "item"_a)
        .def("add_descriptor", &heap_wrapper::add_descriptor, "descriptor"_a)
        .def("add_start", &heap_wrapper::add_start)
        .def("add_end", &heap_wrapper::add_end)
        .def_property("repeat_pointers",
                      &heap_wrapper::get_repeat_pointers,
                      &heap_wrapper::set_repeat_pointers);

    // keep_alive is safe to use here in spite of pybind/pybind11#856, because
    // the destructor of packet_generator doesn't reference the heap.
    py::class_<packet_generator>(m, "PacketGenerator")
        .def(py::init<heap_wrapper &, item_pointer_t, std::size_t>(),
             "heap"_a, "cnt"_a, "max_packet_size"_a,
             py::keep_alive<1, 2>())
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", &packet_generator_next);

    py::enum_<rate_method>(m, "RateMethod")
        .value("SW", rate_method::SW)
        .value("HW", rate_method::HW)
        .value("AUTO", rate_method::AUTO);

    py::enum_<group_mode>(m, "GroupMode")
        .value("ROUND_ROBIN", group_mode::ROUND_ROBIN)
        .value("SERIAL", group_mode::SERIAL);

    py::class_<heap_reference>(m, "HeapReference")
        .def(py::init<const heap_wrapper &, s_item_pointer_t, std::size_t, double>(),
             "heap"_a, py::kw_only(), "cnt"_a = -1, "substream_index"_a = 0, "rate"_a = -1.0,
             py::keep_alive<1, 2>())
        .def_property_readonly(
            "heap",
            [](const heap_reference &h) { return static_cast<const heap_wrapper *>(&h.heap); },
            py::return_value_policy::reference)
        .def_readwrite("cnt", &heap_reference::cnt)
        .def_readwrite("substream_index", &heap_reference::substream_index)
        .def_readwrite("rate", &heap_reference::rate);

    py::class_<heap_reference_list>(m, "HeapReferenceList")
        .def(py::init<std::vector<heap_reference>>(), "heaps"_a)
        .def("__len__", &heap_reference_list::size)
        .def("__getitem__", &heap_reference_list::get_slice);

    py::class_<stream_config>(m, "StreamConfig")
        .def(py::init(&data_class_constructor<stream_config>))
        .def_property("max_packet_size",
                      &stream_config::get_max_packet_size,
                      SPEAD2_PTMF_VOID(stream_config, set_max_packet_size))
        .def_property("rate",
                      &stream_config::get_rate,
                      SPEAD2_PTMF_VOID(stream_config, set_rate))
        .def_property("burst_size",
                      &stream_config::get_burst_size,
                      SPEAD2_PTMF_VOID(stream_config, set_burst_size))
        .def_property("max_heaps",
                      &stream_config::get_max_heaps,
                      SPEAD2_PTMF_VOID(stream_config, set_max_heaps))
        .def_property("burst_rate_ratio",
                      &stream_config::get_burst_rate_ratio,
                      SPEAD2_PTMF_VOID(stream_config, set_burst_rate_ratio))
        .def_property("rate_method",
                      &stream_config::get_rate_method,
                      SPEAD2_PTMF_VOID(stream_config, set_rate_method))
        .def_property_readonly("burst_rate",
                               &stream_config::get_burst_rate)
        .def_readonly_static("DEFAULT_MAX_PACKET_SIZE", &stream_config::default_max_packet_size)
        .def_readonly_static("DEFAULT_MAX_HEAPS", &stream_config::default_max_heaps)
        .def_readonly_static("DEFAULT_BURST_SIZE", &stream_config::default_burst_size)
        .def_readonly_static("DEFAULT_BURST_RATE_RATIO", &stream_config::default_burst_rate_ratio)
        .def_readonly_static("DEFAULT_RATE_METHOD", &stream_config::default_rate_method);

    py::class_<stream>(m, "Stream")
        .def("set_cnt_sequence", &stream::set_cnt_sequence,
             "next"_a, "step"_a)
        .def_property_readonly("num_substreams", &stream::get_num_substreams);

    {
        auto stream_class = udp_stream_register<udp_stream_wrapper<stream_wrapper<udp_stream>>>(m, "UdpStream");
        sync_stream_register(stream_class);
    }
    {
        auto stream_class = udp_stream_register<udp_stream_wrapper<asyncio_stream_wrapper<udp_stream>>>(m, "UdpStreamAsyncio");
        async_stream_register(stream_class);
    }

#if SPEAD2_USE_IBV
    py::class_<udp_ibv_config_wrapper>(m, "UdpIbvConfig")
        .def(py::init(&data_class_constructor<udp_ibv_config_wrapper>))
        .def_readwrite("endpoints", &udp_ibv_config_wrapper::py_endpoints)
        .def_readwrite("memory_regions", &udp_ibv_config_wrapper::py_memory_regions)
        .def_readwrite("interface_address", &udp_ibv_config_wrapper::py_interface_address)
        .def_property("buffer_size",
                      &udp_ibv_config_wrapper::get_buffer_size,
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_buffer_size))
        .def_property("ttl",
                      &udp_ibv_config_wrapper::get_ttl,
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_ttl))
        .def_property("comp_vector",
                      &udp_ibv_config_wrapper::get_comp_vector,
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_comp_vector))
        .def_property("max_poll",
                      &udp_ibv_config_wrapper::get_max_poll,
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_max_poll))
        .def_readonly_static("DEFAULT_BUFFER_SIZE", &udp_ibv_config_wrapper::default_buffer_size)
        .def_readonly_static("DEFAULT_MAX_POLL", &udp_ibv_config_wrapper::default_max_poll);

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
        py::class_<bytes_stream, stream> stream_class(m, "BytesStream", py::multiple_inheritance());
        stream_class
            .def(py::init<std::shared_ptr<thread_pool_wrapper>, const stream_config &>(),
                 "thread_pool"_a.none(false), "config"_a = stream_config())
            .def("getvalue", &bytes_stream::getvalue);
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
