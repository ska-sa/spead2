/* Copyright 2015, 2017, 2020-2023 National Research Foundation (SARAO)
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
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <cstdint>
#include <cctype>
#include <unistd.h>
#include <sys/socket.h>
#include <spead2/recv_udp.h>
#include <spead2/recv_udp_ibv.h>
#include <spead2/recv_udp_pcap.h>
#include <spead2/recv_tcp.h>
#include <spead2/recv_mem.h>
#include <spead2/recv_inproc.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_ring_stream.h>
#include <spead2/recv_chunk_stream.h>
#include <spead2/recv_chunk_stream_group.h>
#include <spead2/recv_live_heap.h>
#include <spead2/recv_heap.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/py_common.h>

namespace py = pybind11;

namespace spead2
{
namespace recv
{

/* True if both arguments are plain function pointers and point to the same function */
template<typename R, typename... Args>
static bool equal_functions(const std::function<R(Args...)> &a,
                            const std::function<R(Args...)> &b)
{
    using ptr = R (*)(Args...);
    const ptr *x = a.template target<ptr>();
    const ptr *y = b.template target<ptr>();
    return x && y && *x == *y;
}

/**
 * Wraps @ref item to provide safe memory management. The item references
 * memory inside the heap, so it needs to hold a reference to that
 * heap, as do any memoryviews created on the value.
 */
class item_wrapper : public item
{
private:
    /// Python object containing a @ref heap
    py::object owning_heap;

public:
    item_wrapper() = default;
    item_wrapper(const item &it, py::object owning_heap)
        : item(it), owning_heap(std::move(owning_heap)) {}

    /**
     * Obtain the raw value using Python buffer protocol.
     */
    py::buffer_info get_value() const
    {
        return py::buffer_info(
            reinterpret_cast<void *>(ptr),
            1,      // size of individual elements
            py::format_descriptor<std::uint8_t>::format(),
            length);
    }
};

/**
 * Extends mem_reader to obtain data using the Python buffer protocol.
 */
class buffer_reader : public mem_reader
{
private:
    py::buffer_info view;
public:
    explicit buffer_reader(stream &s, py::buffer_info view)
        : mem_reader(s, reinterpret_cast<const std::uint8_t *>(view.ptr), view.itemsize * view.size),
        view(std::move(view))
    {
    }
};

#if SPEAD2_USE_IBV
/* Managing the endpoints and interface address requires some sleight of
 * hand. We store a separate copy in the wrapper in a Python-centric format.
 * When constructing the reader, we make a copy with the C++ view.
 */
class udp_ibv_config_wrapper : public udp_ibv_config
{
public:
    std::vector<std::pair<std::string, std::uint16_t>> py_endpoints;
    std::string py_interface_address;
};
#endif // SPEAD2_USE_IBV

static boost::asio::ip::address make_address(stream &s, const std::string &hostname)
{
    return make_address_no_release(s.get_io_service(), hostname,
                                   boost::asio::ip::udp::resolver::query::passive);
}

template<typename Protocol>
static typename Protocol::endpoint make_endpoint(
    stream &s, const std::string &hostname, std::uint16_t port)
{
    return typename Protocol::endpoint(make_address(s, hostname), port);
}

static void add_buffer_reader(stream &s, py::buffer buffer)
{
    py::buffer_info info = request_buffer_info(buffer, PyBUF_C_CONTIGUOUS);
    py::gil_scoped_release gil;
    s.emplace_reader<buffer_reader>(std::move(info));
}

static void add_udp_reader(
    stream &s,
    std::uint16_t port,
    std::size_t max_size,
    std::size_t buffer_size,
    const std::string &bind_hostname)
{
    py::gil_scoped_release gil;
    auto endpoint = make_endpoint<boost::asio::ip::udp>(s, bind_hostname, port);
    s.emplace_reader<udp_reader>(endpoint, max_size, buffer_size);
}

static void add_udp_reader_socket(
    stream &s,
    const socket_wrapper<boost::asio::ip::udp::socket> &socket,
    std::size_t max_size = udp_reader::default_max_size)
{
    auto asio_socket = socket.copy(s.get_io_service());
    py::gil_scoped_release gil;
    s.emplace_reader<udp_reader>(std::move(asio_socket), max_size);
}

static void add_udp_reader_bind_v4(
    stream &s,
    const std::string &address,
    std::uint16_t port,
    std::size_t max_size,
    std::size_t buffer_size,
    const std::string &interface_address)
{
    py::gil_scoped_release gil;
    auto endpoint = make_endpoint<boost::asio::ip::udp>(s, address, port);
    s.emplace_reader<udp_reader>(endpoint, max_size, buffer_size, make_address(s, interface_address));
}

static void add_udp_reader_bind_v6(
    stream &s,
    const std::string &address,
    std::uint16_t port,
    std::size_t max_size,
    std::size_t buffer_size,
    unsigned int interface_index)
{
    py::gil_scoped_release gil;
    auto endpoint = make_endpoint<boost::asio::ip::udp>(s, address, port);
    s.emplace_reader<udp_reader>(endpoint, max_size, buffer_size, interface_index);
}

static void add_tcp_reader(
    stream &s,
    std::uint16_t port,
    std::size_t max_size,
    std::size_t buffer_size,
    const std::string &bind_hostname)
{
    py::gil_scoped_release gil;
    auto endpoint = make_endpoint<boost::asio::ip::tcp>(s, bind_hostname, port);
    s.emplace_reader<tcp_reader>(endpoint, max_size, buffer_size);
}

static void add_tcp_reader_socket(
    stream &s,
    const socket_wrapper<boost::asio::ip::tcp::acceptor> &acceptor,
    std::size_t max_size)
{
    auto asio_socket = acceptor.copy(s.get_io_service());
    py::gil_scoped_release gil;
    s.emplace_reader<tcp_reader>(std::move(asio_socket), max_size);
}

#if SPEAD2_USE_IBV
static void add_udp_ibv_reader(stream &s, const udp_ibv_config_wrapper &config_wrapper)
{
    py::gil_scoped_release gil;
    udp_ibv_config config = config_wrapper;
    for (const auto &[host, port] : config_wrapper.py_endpoints)
        config.add_endpoint(make_endpoint<boost::asio::ip::udp>(s, host, port));
    config.set_interface_address(
        make_address(s, config_wrapper.py_interface_address));
    s.emplace_reader<udp_ibv_reader>(config);
}
#endif  // SPEAD2_USE_IBV

#if SPEAD2_USE_PCAP
static void add_udp_pcap_file_reader(stream &s, const std::string &filename, const std::string &filter)
{
    py::gil_scoped_release gil;
    s.emplace_reader<udp_pcap_file_reader>(filename, filter);
}
#endif

static void add_inproc_reader(stream &s, std::shared_ptr<inproc_queue> queue)
{
    py::gil_scoped_release gil;
    s.emplace_reader<inproc_reader>(queue);
}

class ring_stream_config_wrapper : public ring_stream_config
{
private:
    bool incomplete_keep_payload_ranges = false;

public:
    ring_stream_config_wrapper() = default;

    ring_stream_config_wrapper(const ring_stream_config &base) :
        ring_stream_config(base)
    {
    }

    ring_stream_config_wrapper &set_incomplete_keep_payload_ranges(bool keep)
    {
        incomplete_keep_payload_ranges = keep;
        return *this;
    }

    bool get_incomplete_keep_payload_ranges() const
    {
        return incomplete_keep_payload_ranges;
    }
};

/**
 * Stream that handles the magic necessary to reflect heaps into
 * Python space and capture the reference to it.
 *
 * The GIL needs to be handled carefully. Any operation run by the thread pool
 * might need to take the GIL to do logging. Thus, any operation that blocks
 * on completion of code scheduled through the thread pool must drop the GIL
 * first.
 */
class ring_stream_wrapper : public ring_stream<ringbuffer<live_heap, semaphore_fd, semaphore> >
{
private:
    bool incomplete_keep_payload_ranges;
    exit_stopper stopper{[this] { stop(); }};

    py::object to_object(live_heap &&h)
    {
        if (h.is_contiguous())
            return py::cast(heap(std::move(h)), py::return_value_policy::move);
        else
            return py::cast(incomplete_heap(std::move(h), false, incomplete_keep_payload_ranges),
                            py::return_value_policy::move);
    }

public:
    ring_stream_wrapper(
        io_service_ref io_service,
        const stream_config &config = stream_config(),
        const ring_stream_config_wrapper &ring_config = ring_stream_config_wrapper())
        : ring_stream<ringbuffer<live_heap, semaphore_fd, semaphore>>(
            std::move(io_service), config, ring_config),
        incomplete_keep_payload_ranges(ring_config.get_incomplete_keep_payload_ranges())
    {}

    py::object next()
    {
        try
        {
            return get();
        }
        catch (ringbuffer_stopped &e)
        {
            throw py::stop_iteration();
        }
    }

    py::object get()
    {
        return to_object(ring_stream::pop_live(gil_release_tag()));
    }

    py::object get_nowait()
    {
        return to_object(try_pop_live());
    }

    int get_fd() const
    {
        return get_ringbuffer().get_data_sem().get_fd();
    }

    ring_stream_config_wrapper get_ring_config() const
    {
        ring_stream_config_wrapper ring_config(
            ring_stream<ringbuffer<live_heap, semaphore_fd, semaphore> >::get_ring_config());
        ring_config.set_incomplete_keep_payload_ranges(incomplete_keep_payload_ranges);
        return ring_config;
    }

    virtual void stop() override
    {
        stopper.reset();
        py::gil_scoped_release gil;
        ring_stream::stop();
    }

    ~ring_stream_wrapper()
    {
        stop();
    }
};

/**
 * Package a chunk with a reference to the original Python object.
 * This is used
 * only for chunks in the ringbuffer, not those owned by Python.
 */
class chunk_wrapper : public chunk
{
public:
    py::object obj;
};

/**
 * Get the original Python object from a wrapped chunk, and
 * restore its pointers.
 */
static py::object unwrap_chunk(std::unique_ptr<chunk> &&c)
{
    chunk_wrapper &cw = dynamic_cast<chunk_wrapper &>(*c);
    chunk &orig = cw.obj.cast<chunk &>();
    py::object ret = std::move(cw.obj);
    orig = std::move(*c);
    return ret;
}

/**
 * Wrap up a Python chunk into an object that can traverse the ringbuffer.
 * Python doesn't allow ownership to be given away, so we have to create a
 * new C++ object which refers back to the original Python object to keep
 * it alive.
 */
static std::unique_ptr<chunk_wrapper> wrap_chunk(chunk &c)
{
    if (!c.data)
        throw std::invalid_argument("data buffer is not set");
    if (!c.present)
        throw std::invalid_argument("present buffer is not set");
    auto cw = std::make_unique<chunk_wrapper>();
    static_cast<chunk &>(*cw) = std::move(c);
    cw->obj = py::cast(c);
    return cw;
}

/**
 * Push a chunk onto a ringbuffer. The specific operation is described by
 * @a func; this function takes care of wrapping in @ref chunk_wrapper.
 */
template<typename T>
static void push_chunk(T func, chunk &c)
{
    /* Note: the type of 'wrapper' must exactly match what the ringbuffer
     * expects, otherwise it constructs a new, temporary unique_ptr by
     * moving from 'wrapper', and we lose ownership in the failure path.
     */
    std::unique_ptr<chunk> wrapper = wrap_chunk(c);
    try
    {
        func(std::move(wrapper));
    }
    catch (std::exception &)
    {
        // Undo the move that happened as part of wrapping
        if (wrapper)
            c = std::move(*wrapper);
        throw;
    }
}

typedef ringbuffer<std::unique_ptr<chunk>, semaphore_fd, semaphore_fd> chunk_ringbuffer;

/* Note: ring_stream_wrapper drops the GIL while stopping. We
 * can't do that here because stop() can free chunks that were
 * in flight, which involves interaction with the Python API.
 * I think the only reason ring_stream_wrapper drops the GIL is
 * that logging used to directly acquire the GIL, and so if stop()
 * did any logging it would deadlock. Now that logging is pushed
 * off to a separate thread that should no longer be an issue.
 */
#define EXIT_STOPPER_WRAPPER(cls, base)                   \
    class cls : public base                               \
    {                                                     \
    private:                                              \
        exit_stopper stopper{[this] { stop(); }};         \
    public:                                               \
        using base::base;                                 \
        virtual void stop() override                      \
        {                                                 \
            stopper.reset();                              \
            base::stop();                                 \
        }                                                 \
    }

// These aliases are needed because a type passed to a macro cannot contain a comma
using chunk_ring_stream_orig = chunk_ring_stream<chunk_ringbuffer, chunk_ringbuffer>;
using chunk_stream_ring_group_orig = chunk_stream_ring_group<chunk_ringbuffer, chunk_ringbuffer>;

EXIT_STOPPER_WRAPPER(chunk_ring_stream_wrapper, chunk_ring_stream_orig);
EXIT_STOPPER_WRAPPER(chunk_stream_ring_group_wrapper, chunk_stream_ring_group_orig);
EXIT_STOPPER_WRAPPER(chunk_stream_group_member_wrapper, chunk_stream_group_member);

#undef EXIT_STOPPER_WRAPPER

/// Register the receiver module with Python
py::module register_module(py::module &parent)
{
    using namespace pybind11::literals;

    // Create the module
    py::module m = parent.def_submodule("recv");

    py::class_<heap_base>(m, "HeapBase")
        .def_property_readonly("cnt", &heap_base::get_cnt)
        .def_property_readonly("flavour", &heap_base::get_flavour)
        .def("get_items", [](py::object &self) -> py::list
        {
            const heap_base &h = self.cast<const heap_base &>();
            std::vector<item> base = h.get_items();
            py::list out;
            for (const item &it : base)
            {
                // Filter out descriptors here. The base class can't do so, because
                // the descriptors are retrieved from the items.
                if (it.id != DESCRIPTOR_ID)
                    out.append(item_wrapper(it, self));
            }
            return out;
        })
        .def("is_start_of_stream", &heap_base::is_start_of_stream)
        .def("is_end_of_stream", &heap_base::is_end_of_stream);
    py::class_<heap, heap_base>(m, "Heap")
        .def("get_descriptors", &heap::get_descriptors);
    py::class_<incomplete_heap, heap_base>(m, "IncompleteHeap")
        .def_property_readonly("heap_length", &incomplete_heap::get_heap_length)
        .def_property_readonly("received_length", &incomplete_heap::get_received_length)
        .def_property_readonly("payload_ranges", &incomplete_heap::get_payload_ranges);
    py::class_<item_wrapper>(m, "RawItem", py::buffer_protocol())
        .def_readonly("id", &item_wrapper::id)
        .def_readonly("is_immediate", &item_wrapper::is_immediate)
        .def_readonly("immediate_value", &item_wrapper::immediate_value)
        .def_buffer([](item_wrapper &item) { return item.get_value(); });

    py::class_<stream_stat_config> stream_stat_config_cls(m, "StreamStatConfig");
    /* We have to register the embedded enum type before we can use it as a
     * default value for the stream_stat constructor/
     */
    py::enum_<stream_stat_config::mode>(stream_stat_config_cls, "Mode")
        .value("COUNTER", stream_stat_config::mode::COUNTER)
        .value("MAXIMUM", stream_stat_config::mode::MAXIMUM);
    stream_stat_config_cls
        .def(
            py::init<std::string, stream_stat_config::mode>(),
            "name"_a, "mode"_a = stream_stat_config::mode::COUNTER)
        .def_property_readonly("name", &stream_stat_config::get_name)
        .def_property_readonly("mode", &stream_stat_config::get_mode)
        .def("combine", &stream_stat_config::combine)
        .def(py::self == py::self)
        .def(py::self != py::self);
    py::class_<stream_stats> stream_stats_cls(m, "StreamStats");
    stream_stats_cls
        .def("__getitem__", [](const stream_stats &self, std::size_t index)
        {
            if (index < self.size())
                return self[index];
            else
                throw py::index_error();
        })
        .def("__getitem__", [](const stream_stats &self, const std::string &name)
        {
            auto pos = self.find(name);
            if (pos == self.end())
                throw py::key_error(name);
            return pos->second;
        })
        .def("__setitem__", [](stream_stats &self, std::size_t index, std::uint64_t value)
        {
            if (index < self.size())
                self[index] = value;
            else
                throw py::index_error();
        })
        .def("__setitem__", [](stream_stats &self, const std::string &name, std::uint64_t value)
        {
            auto pos = self.find(name);
            if (pos == self.end())
                throw py::key_error(name);
            pos->second = value;
        })
        .def("__contains__", [](const stream_stats &self, const std::string &name)
        {
            return self.find(name) != self.end();
        })
        .def("get", [](const stream_stats &self, const std::string &name, py::object &default_)
        {
            auto pos = self.find(name);
            return pos != self.end() ? py::int_(pos->second) : default_;
        }, py::arg(), py::arg() = py::none())
        /* TODO: keys, values and items should ideally return view that
         * simulate Python's dictionary views (py::bind_map does this, but it
         * can't be used because it expects the map to implement erase).
         */
        .def(
            "items",
            [](const stream_stats &self) { return py::make_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>()  // keep the stats alive while it is iterated
        )
        .def(
            "__iter__",
            [](const stream_stats &self) { return py::make_key_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>()  // keep the stats alive while it is iterated
        )
        .def(
            "keys",
            [](const stream_stats &self) { return py::make_key_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>()  // keep the stats alive while it is iterated
        )
        .def(
            "values",
            [](const stream_stats &self) { return py::make_value_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>()  // keep the stats alive while it is iterated
        )
        .def("__len__", &stream_stats::size)
        .def_property_readonly("config", &stream_stats::get_config)
        .def(py::self + py::self)
        .def(py::self += py::self);

    py::module stream_stat_indices_module = m.def_submodule("stream_stat_indices");
    /* The macro registers a property on stream_stats to access the built-in stats
     * by name, and at the same time populates the index constant in submodule
     * stream_stat_indices (upper-casing it).
     */
#define STREAM_STATS_PROPERTY(field) \
    do { \
        stream_stats_cls.def_property( \
            #field, \
            [](const stream_stats &self) { return self[stream_stat_indices::field]; }, \
            [](stream_stats &self, std::uint64_t value) { self[stream_stat_indices::field] = value; }); \
        std::string upper = #field; \
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper); \
        stream_stat_indices_module.attr(upper.c_str()) = stream_stat_indices::field; \
    } while (false)

    STREAM_STATS_PROPERTY(heaps);
    STREAM_STATS_PROPERTY(incomplete_heaps_evicted);
    STREAM_STATS_PROPERTY(incomplete_heaps_flushed);
    STREAM_STATS_PROPERTY(packets);
    STREAM_STATS_PROPERTY(batches);
    STREAM_STATS_PROPERTY(worker_blocked);
    STREAM_STATS_PROPERTY(max_batch);
    STREAM_STATS_PROPERTY(single_packet_heaps);
    STREAM_STATS_PROPERTY(search_dist);
#undef STREAM_STATS_PROPERTY

    py::class_<stream_config>(m, "StreamConfig")
        .def(py::init(&data_class_constructor<stream_config>))
        .def_property("max_heaps",
                      &stream_config::get_max_heaps,
                      &stream_config::set_max_heaps)
        .def_property("substreams",
                      &stream_config::get_substreams,
                      &stream_config::set_substreams)
        .def_property("bug_compat",
                      &stream_config::get_bug_compat,
                      &stream_config::set_bug_compat)
        .def_property("memcpy",
             [](const stream_config &self) {
                 stream_config cmp;
                 memcpy_function_id ids[] = {MEMCPY_STD, MEMCPY_NONTEMPORAL};
                 for (memcpy_function_id id : ids)
                 {
                     cmp.set_memcpy(id);
                     if (equal_functions(self.get_memcpy(), cmp.get_memcpy()))
                         return int(id);
                 }
                 throw std::invalid_argument("memcpy function is not one of the standard ones");
             },
             [](stream_config &self, int id) { self.set_memcpy(memcpy_function_id(id)); })
        .def_property("memory_allocator",
                      &stream_config::get_memory_allocator,
                      SPEAD2_PTMF_VOID(stream_config, set_memory_allocator))
        .def_property("stop_on_stop_item",
                      &stream_config::get_stop_on_stop_item,
                      SPEAD2_PTMF_VOID(stream_config, set_stop_on_stop_item))
        .def_property("allow_unsized_heaps",
                      &stream_config::get_allow_unsized_heaps,
                      SPEAD2_PTMF_VOID(stream_config, set_allow_unsized_heaps))
        .def_property("allow_out_of_order",
                      &stream_config::get_allow_out_of_order,
                      SPEAD2_PTMF_VOID(stream_config, set_allow_out_of_order))
        .def_property("stream_id",
                      &stream_config::get_stream_id,
                      &stream_config::set_stream_id)
        .def("add_stat", &stream_config::add_stat,
             "name"_a,
             "mode"_a = stream_stat_config::mode::COUNTER)
        .def_property_readonly("stats", &stream_config::get_stats)
        .def("get_stat_index", &stream_config::get_stat_index,
             "name"_a)
        .def("next_stat_index", &stream_config::next_stat_index)
        .def_readonly_static("DEFAULT_MAX_HEAPS", &stream_config::default_max_heaps);
    py::class_<ring_stream_config_wrapper>(m, "RingStreamConfig")
        .def(py::init(&data_class_constructor<ring_stream_config_wrapper>))
        .def_property("heaps",
                      &ring_stream_config_wrapper::get_heaps,
                      SPEAD2_PTMF_VOID(ring_stream_config_wrapper, set_heaps))
        .def_property("contiguous_only",
                      &ring_stream_config_wrapper::get_contiguous_only,
                      SPEAD2_PTMF_VOID(ring_stream_config_wrapper, set_contiguous_only))
        .def_property("incomplete_keep_payload_ranges",
                      &ring_stream_config_wrapper::get_incomplete_keep_payload_ranges,
                      SPEAD2_PTMF_VOID(ring_stream_config_wrapper, set_incomplete_keep_payload_ranges))
        .def_readonly_static("DEFAULT_HEAPS", &ring_stream_config_wrapper::default_heaps);
#if SPEAD2_USE_IBV
    py::class_<udp_ibv_config_wrapper>(m, "UdpIbvConfig")
        .def(py::init(&data_class_constructor<udp_ibv_config_wrapper>))
        .def_readwrite("endpoints", &udp_ibv_config_wrapper::py_endpoints)
        .def_readwrite("interface_address", &udp_ibv_config_wrapper::py_interface_address)
        .def_property("buffer_size",
                      &udp_ibv_config_wrapper::get_buffer_size,
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_buffer_size))
        .def_property("max_size",
                      &udp_ibv_config_wrapper::get_max_size,
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_max_size))
        .def_property("comp_vector",
                      &udp_ibv_config_wrapper::get_comp_vector,
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_comp_vector))
        .def_property("max_poll",
                      &udp_ibv_config_wrapper::get_max_poll,
                      SPEAD2_PTMF_VOID(udp_ibv_config_wrapper, set_max_poll))
        .def_readonly_static("DEFAULT_BUFFER_SIZE", &udp_ibv_config_wrapper::default_buffer_size)
        .def_readonly_static("DEFAULT_MAX_SIZE", &udp_ibv_config_wrapper::default_max_size)
        .def_readonly_static("DEFAULT_MAX_POLL", &udp_ibv_config_wrapper::default_max_poll);
#endif // SPEAD2_USE_IBV
    py::class_<stream>(m, "_Stream")
        // SPEAD2_PTMF doesn't work for get_stats because it's defined in stream_base, which is a protected ancestor
        .def_property_readonly("stats", [](const stream &self) { return self.get_stats(); })
        .def_property_readonly("config",
                               [](const stream &self) { return self.get_config(); })
        .def("add_buffer_reader", add_buffer_reader, "buffer"_a)
        .def("add_udp_reader", add_udp_reader,
              "port"_a,
              "max_size"_a = udp_reader::default_max_size,
              "buffer_size"_a = udp_reader::default_buffer_size,
              "bind_hostname"_a = std::string())
        .def("add_udp_reader", add_udp_reader_socket,
              "socket"_a,
              "max_size"_a = udp_reader::default_max_size)
        .def("add_udp_reader", add_udp_reader_bind_v4,
              "multicast_group"_a,
              "port"_a,
              "max_size"_a = udp_reader::default_max_size,
              "buffer_size"_a = udp_reader::default_buffer_size,
              "interface_address"_a = "0.0.0.0")
        .def("add_udp_reader", add_udp_reader_bind_v6,
              "multicast_group"_a,
              "port"_a,
              "max_size"_a = udp_reader::default_max_size,
              "buffer_size"_a = udp_reader::default_buffer_size,
              "interface_index"_a = (unsigned int) 0)
        .def("add_tcp_reader", add_tcp_reader,
             "port"_a,
             "max_size"_a = tcp_reader::default_max_size,
             "buffer_size"_a = tcp_reader::default_buffer_size,
             "bind_hostname"_a = std::string())
        .def("add_tcp_reader", add_tcp_reader_socket,
             "acceptor"_a,
             "max_size"_a = tcp_reader::default_max_size)
#if SPEAD2_USE_IBV
        .def("add_udp_ibv_reader", add_udp_ibv_reader,
             "config"_a)
#endif
#if SPEAD2_USE_PCAP
        .def("add_udp_pcap_file_reader", add_udp_pcap_file_reader,
             "filename"_a, "filter"_a = "")
#endif
        .def("add_inproc_reader", add_inproc_reader,
             "queue"_a)
        .def("stop", &stream::stop)
        .def_readonly_static("DEFAULT_UDP_MAX_SIZE", &udp_reader::default_max_size)
        .def_readonly_static("DEFAULT_UDP_BUFFER_SIZE", &udp_reader::default_buffer_size)
        .def_readonly_static("DEFAULT_TCP_MAX_SIZE", &tcp_reader::default_max_size)
        .def_readonly_static("DEFAULT_TCP_BUFFER_SIZE", &tcp_reader::default_buffer_size);
    py::class_<ring_stream_wrapper, stream> stream_class(m, "Stream");
    stream_class
        .def(py::init<std::shared_ptr<thread_pool_wrapper>,
                      const stream_config &,
                      const ring_stream_config_wrapper &>(),
             "thread_pool"_a.none(false), "config"_a = stream_config(),
             "ring_config"_a = ring_stream_config_wrapper())
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", &ring_stream_wrapper::next)
        .def("get", &ring_stream_wrapper::get)
        .def("get_nowait", &ring_stream_wrapper::get_nowait)
        .def_property_readonly("fd", &ring_stream_wrapper::get_fd)
        .def_property_readonly("ringbuffer", &ring_stream_wrapper::get_ringbuffer)
        .def_property_readonly("ring_config", &ring_stream_wrapper::get_ring_config);
    using Ringbuffer = ringbuffer<live_heap, semaphore_fd, semaphore>;
    py::class_<Ringbuffer>(stream_class, "Ringbuffer")
        .def("size", &Ringbuffer::size)
        .def("capacity", &Ringbuffer::capacity);
    py::class_<chunk_stream_config>(m, "ChunkStreamConfig")
        .def(py::init(&data_class_constructor<chunk_stream_config>))
        .def_property("items",
                      &chunk_stream_config::get_items,
                      &chunk_stream_config::set_items)
        .def_property("max_chunks",
                      &chunk_stream_config::get_max_chunks,
                      &chunk_stream_config::set_max_chunks)
        .def_property(
            "place",
            [](const chunk_stream_config &config) {
                return callback_to_python(config.get_place());
            },
            [](chunk_stream_config &config, py::object obj) {
                config.set_place(callback_from_python<chunk_place_function>(
                    obj,
                    "void (void *, size_t)",
                    "void (void *, size_t, void *)"
                ));
            })
        .def(
            "enable_packet_presence", &chunk_stream_config::enable_packet_presence,
            "payload_size"_a)
        .def("disable_packet_presence", &chunk_stream_config::disable_packet_presence)
        .def_property_readonly("packet_presence_payload_size",
                               &chunk_stream_config::get_packet_presence_payload_size)
        .def_property("max_heap_extra",
                      &chunk_stream_config::get_max_heap_extra,
                      &chunk_stream_config::set_max_heap_extra)
        .def_readonly_static("DEFAULT_MAX_CHUNKS", &chunk_stream_config::default_max_chunks);
    py::class_<chunk>(m, "Chunk")
        .def(py::init(&data_class_constructor<chunk>))
        .def_readwrite("chunk_id", &chunk::chunk_id)
        .def_readwrite("stream_id", &chunk::stream_id)
        // Can't use def_readwrite for present, data, extra because they're
        // non-copyable types
        .def_property(
            "present",
            [](const chunk &c) -> const memory_allocator::pointer & { return c.present; },
            [](chunk &c, memory_allocator::pointer &&value)
            {
                if (value)
                {
                    auto *alloc = get_buffer_allocation(value);
                    assert(alloc != nullptr);
                    c.present_size = alloc->buffer_info.size * alloc->buffer_info.itemsize;
                }
                else
                    c.present_size = 0;
                c.present = std::move(value);
            })
        .def_property(
            "data",
            [](const chunk &c) -> const memory_allocator::pointer & { return c.data; },
            [](chunk &c, memory_allocator::pointer &&value) { c.data = std::move(value); })
        .def_property(
            "extra",
            [](const chunk &c) -> const memory_allocator::pointer & { return c.extra; },
            [](chunk &c, memory_allocator::pointer &&value) { c.extra = std::move(value); });
    // Don't allow ChunkRingPair to be constructed from Python. It exists
    // purely to be a base class.
    using chunk_ring_pair = detail::chunk_ring_pair<chunk_ringbuffer, chunk_ringbuffer>;
    py::class_<chunk_ring_pair>(m, "ChunkRingPair")
        .def(
            "add_free_chunk",
            [](chunk_ring_pair &self, chunk &c)
            {
                push_chunk(
                    [&self](std::unique_ptr<chunk> &&wrapper)
                    {
                        self.add_free_chunk(std::move(wrapper));
                    },
                    c
                );
            },
            "chunk"_a)
        .def_property_readonly("data_ringbuffer", &chunk_ring_pair::get_data_ringbuffer)
        .def_property_readonly("free_ringbuffer", &chunk_ring_pair::get_free_ringbuffer);

    py::class_<chunk_ring_stream_wrapper,
               detail::chunk_ring_pair<chunk_ringbuffer, chunk_ringbuffer>,
               stream>(m, "ChunkRingStream")
        .def(py::init<std::shared_ptr<thread_pool_wrapper>,
                      const stream_config &,
                      const chunk_stream_config &,
                      std::shared_ptr<chunk_ringbuffer>,
                      std::shared_ptr<chunk_ringbuffer>>(),
             "thread_pool"_a.none(false),
             "config"_a = stream_config(),
             "chunk_stream_config"_a,
             "data_ringbuffer"_a.none(false),
             "free_ringbuffer"_a.none(false),
            // Keep the Python ringbuffer objects alive, not just the C++ side.
            // This allows Python subclasses to be passed then later retrieved
            // from properties.
             py::keep_alive<1, 5>(),
             py::keep_alive<1, 6>());
    py::class_<chunk_ringbuffer, std::shared_ptr<chunk_ringbuffer>>(m, "ChunkRingbuffer")
        .def(py::init<std::size_t>(), "maxsize"_a)
        .def("qsize", &chunk_ringbuffer::size)
        .def_property_readonly("maxsize", &chunk_ringbuffer::capacity)
        .def_property_readonly(
            "data_fd",
            [](const chunk_ringbuffer &ring) { return ring.get_data_sem().get_fd(); })
        .def_property_readonly(
            "space_fd",
            [](const chunk_ringbuffer &ring) { return ring.get_space_sem().get_fd(); })
        .def("get", [](chunk_ringbuffer &ring) { return unwrap_chunk(ring.pop(gil_release_tag())); })
        .def("get_nowait", [](chunk_ringbuffer &ring) { return unwrap_chunk(ring.try_pop()); })
        .def(
            "put",
            [](chunk_ringbuffer &ring, chunk &c)
            {
                push_chunk(
                    [&ring](std::unique_ptr<chunk> &&wrapper)
                    {
                        ring.push(std::move(wrapper), gil_release_tag());
                    },
                    c
                );
            },
            "chunk"_a)
        .def(
            "put_nowait",
            [](chunk_ringbuffer &ring, chunk &c)
            {
                push_chunk(
                    [&ring](std::unique_ptr<chunk> &&wrapper) { ring.try_push(std::move(wrapper)); },
                    c
                );
            },
            "chunk"_a)
        .def("empty", [](const chunk_ringbuffer &ring) { return ring.size() == 0; })
        .def("full", [](const chunk_ringbuffer &ring) { return ring.size() == ring.capacity(); })
        .def("stop", &chunk_ringbuffer::stop)
        .def("add_producer", &chunk_ringbuffer::add_producer)
        .def("remove_producer", &chunk_ringbuffer::remove_producer)
        .def("__iter__", [](py::object self) { return self; })
        .def(
            "__next__", [](chunk_ringbuffer &ring)
            {
                try
                {
                    return unwrap_chunk(ring.pop(gil_release_tag()));
                }
                catch (ringbuffer_stopped &)
                {
                    throw py::stop_iteration();
                }
            });

    py::class_<chunk_stream_group_config> chunk_stream_group_config_cls(m, "ChunkStreamGroupConfig");
    chunk_stream_group_config_cls
        .def(py::init(&data_class_constructor<chunk_stream_group_config>))
        .def_property("max_chunks",
                      &chunk_stream_group_config::get_max_chunks,
                      &chunk_stream_group_config::set_max_chunks)
        .def_property("eviction_mode",
                      &chunk_stream_group_config::get_eviction_mode,
                      &chunk_stream_group_config::set_eviction_mode)
        .def_readonly_static("DEFAULT_MAX_CHUNKS", &chunk_stream_group_config::default_max_chunks);
    py::enum_<chunk_stream_group_config::eviction_mode>(chunk_stream_group_config_cls, "EvictionMode")
        .value("LOSSY", chunk_stream_group_config::eviction_mode::LOSSY)
        .value("LOSSLESS", chunk_stream_group_config::eviction_mode::LOSSLESS);

    py::class_<chunk_stream_group_member, stream>(m, "ChunkStreamGroupMember");

    py::class_<chunk_stream_ring_group_wrapper,
               detail::chunk_ring_pair<chunk_ringbuffer, chunk_ringbuffer>>(m, "ChunkStreamRingGroup")
        .def(py::init<const chunk_stream_group_config &,
                      std::shared_ptr<chunk_ringbuffer>,
                      std::shared_ptr<chunk_ringbuffer>>(),
             "config"_a,
             "data_ringbuffer"_a.none(false),
             "free_ringbuffer"_a.none(false),
            // Keep the Python ringbuffer objects alive, not just the C++ side.
            // This allows Python subclasses to be passed then later retrieved
            // from properties.
            py::keep_alive<1, 3>(),
            py::keep_alive<1, 4>())
        .def_property_readonly(
            "config", &chunk_stream_ring_group_wrapper::get_config)
        .def(
            "emplace_back",
            [](chunk_stream_ring_group_wrapper &group,
               std::shared_ptr<thread_pool_wrapper> thread_pool,
               const stream_config &config,
               const chunk_stream_config &chunk_stream_config) -> chunk_stream_group_member & {
                return group.emplace_back<chunk_stream_group_member_wrapper>(std::move(thread_pool), config, chunk_stream_config);
            },
            "thread_pool"_a, "config"_a, "chunk_stream_config"_a,
            py::return_value_policy::reference_internal
        )
        .def("__len__", &chunk_stream_ring_group_wrapper::size)
        .def(
            "__getitem__",
            [](chunk_stream_ring_group_wrapper &group, std::ptrdiff_t index) -> chunk_stream_group_member & {
                if (index < 0)
                    index += group.size();
                if (index >= 0 && std::size_t(index) < group.size())
                    return group[index];
                else
                    throw py::index_error();
            },
            py::return_value_policy::reference_internal
        )
        .def(
            "__getitem__",
            [](chunk_stream_ring_group_wrapper &group, const py::slice &slice) {
                py::list out;
                std::size_t start, stop, step, length;
                if (!slice.compute(group.size(), &start, &stop, &step, &length))
                    throw py::error_already_set();
                py::object self = py::cast(group);
                for (std::size_t i = 0; i < length; i++) {
                    out.append(py::cast(group[start], py::return_value_policy::reference_internal, self));
                    start += step;
                }
                return out;
            }
        )
        .def("stop", &chunk_stream_ring_group_wrapper::stop);

    return m;
}

} // namespace recv
} // namespace spead2
