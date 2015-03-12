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
#include "send_heap.h"
#include "send_stream.h"
#include "send_udp.h"
#include "common_thread_pool.h"
#include "py_common.h"

namespace py = boost::python;

namespace spead
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
};

void heap_wrapper::add_item(py::object item)
{
    std::int64_t id = py::extract<std::int64_t>(item.attr("id"));
    py::object buffer = item.attr("to_buffer")();
    bool is_variable_size = py::extract<bool>(item.attr("is_variable_size")());
    item_buffers.emplace_back(buffer);
    const auto &view = item_buffers.back().view;
    heap::add_item(id, view.buf, view.len, !is_variable_size);
}

void heap_wrapper::add_descriptor(py::object object)
{
    heap::add_descriptor(py::extract<descriptor>(object.attr("to_raw")(get_bug_compat())));
}

/**
 * packet_generator takes a reference to an existing basic_heap, but we don't
 * have one. Thus, we need to store one internally.
 */
class packet_generator_wrapper
{
private:
    basic_heap internal_heap;
    packet_generator gen;
public:
    packet_generator_wrapper(const heap &h, int heap_address_bits, std::size_t max_packet_size);

    bytestring next();
};

packet_generator_wrapper::packet_generator_wrapper(
    const heap &h, int heap_address_bits,
    std::size_t max_packet_size)
    : internal_heap(h.encode(heap_address_bits)),
    gen(internal_heap, heap_address_bits, max_packet_size)
{
}

bytestring packet_generator_wrapper::next()
{
    packet pkt = gen.next_packet();
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
        release_gil gil;
        std::promise<void> sent_promise;
        Base::async_send_heap(h, [&sent_promise]()
        {
            sent_promise.set_value();
        });
        sent_promise.get_future().get();
    }
};

template<typename Base>
class asyncio_stream_wrapper : public Base
{
private:
    semaphore_gil sem;
    std::vector<py::object> callbacks;
    std::mutex callbacks_mutex;
public:
    using Base::Base;

    int get_fd() const { return sem.get_fd(); }

    void async_send_heap(py::object h, py::object callback)
    {
        // Note that while h isn't used in the lambda, it is
        // bound to it so that its lifetime persists.
        py::extract<heap_wrapper &> h2(h);
        Base::async_send_heap(h2(), [this, callback, h] () mutable
        {
            {
                std::unique_lock<std::mutex> lock(callbacks_mutex);
                callbacks.push_back(std::move(callback));
            }
            sem.put();
        });
    }

    void process_callbacks()
    {
        sem.get();
        std::vector<py::object> current_callbacks;
        {
            std::unique_lock<std::mutex> lock(callbacks_mutex);
            current_callbacks.swap(callbacks);
        }
        for (const py::object &callback : current_callbacks)
        {
            callback();
        }
    }
};

template<typename Base>
class udp_stream_wrapper : public Base
{
private:
    static boost::asio::ip::udp::endpoint make_endpoint(
        boost::asio::io_service &io_service, const std::string &hostname, int port);

public:
    udp_stream_wrapper(
        thread_pool &pool,
        const std::string &hostname,
        int port,
        int heap_address_bits,
        std::size_t max_packet_size,
        double rate,
        std::size_t max_heaps = Base::default_max_heaps,
        std::size_t buffer_size = Base::default_buffer_size)
        : Base(
            pool.get_io_service(),
            make_endpoint(pool.get_io_service(), hostname, port),
            heap_address_bits, max_packet_size, rate,
            max_heaps, buffer_size)
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

/// Register the send module with Boost.Python
void register_module()
{
    using namespace boost::python;
    using namespace spead::send;

    // Create the module, and set it as the current boost::python scope so that
    // classes we define are added to this module rather than the root.
    py::object module(py::handle<>(py::borrowed(PyImport_AddModule("spead2._send"))));
    py::scope scope = module;

    class_<heap_wrapper, boost::noncopyable>("Heap", init<std::int64_t, bug_compat_mask>(
            (arg("cnt") = 0, arg("bug_compat") = 0)))
        .add_property("cnt", &heap_wrapper::get_cnt, &heap_wrapper::set_cnt)
        .add_property("bug_compat", &heap_wrapper::get_bug_compat, &heap_wrapper::set_bug_compat)
        .def("add_item", &heap_wrapper::add_item,
             arg("item"),
             with_custodian_and_ward<1, 2>())
        .def("add_descriptor", &heap_wrapper::add_descriptor,
             (arg("descriptor")));

    class_<packet_generator_wrapper, boost::noncopyable>("PacketGenerator", init<heap_wrapper &, int, std::size_t>(
            (arg("heap"), arg("heap_address_bits"), arg("max_packet_size")))[
            with_custodian_and_ward<1, 2>()])
        .def("__iter__", objects::identity_function())
        .def(
#if PY_MAJOR_VERSION >= 3
              // Python 3 uses __next__ for the iterator protocol
              "__next__"
#else
              "next"
#endif
              , &packet_generator_wrapper::next);

    {
        typedef udp_stream_wrapper<stream_wrapper<udp_stream> > T;
        class_<T, boost::noncopyable>("UdpStream", init<
                thread_pool_wrapper &, std::string, int, int,
                std::size_t, double, std::size_t, std::size_t>(
                    (arg("thread_pool"), arg("hostname"), arg("port"),
                     arg("heap_address_bits"),
                     arg("max_packet_size"),
                     arg("rate"),
                     arg("max_heaps") = T::default_max_heaps,
                     arg("buffer_size") = T::default_buffer_size))[
                    with_custodian_and_ward<1, 2>()])
            .def("send_heap", &T::send_heap, arg("heap"));
    }

    {
        typedef udp_stream_wrapper<asyncio_stream_wrapper<udp_stream> > T;
        class_<T, boost::noncopyable>("UdpStreamAsyncio", init<
                thread_pool_wrapper &, std::string, int, int,
                std::size_t, double, std::size_t>(
                    (arg("thread_pool"), arg("hostname"), arg("port"),
                     arg("heap_address_bits"),
                     arg("max_packet_size"),
                     arg("rate"),
                     arg("max_heaps") = T::default_max_heaps,
                     arg("buffer_size") = T::default_buffer_size))[
                    with_custodian_and_ward<1, 2>()])
            .add_property("fd", &T::get_fd)
            .def("async_send_heap", &T::async_send_heap, arg("heap"))
            .def("flush", &T::flush)
            .def("process_callbacks", &T::process_callbacks);
    }
}

} // namespace send
} // namespace spead
