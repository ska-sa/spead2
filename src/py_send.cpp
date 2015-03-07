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
    bug_compat_mask bug_compat;
    std::vector<buffer_view> item_buffers;

public:
    explicit heap_wrapper(std::int64_t heap_cnt = 0, bug_compat_mask bug_compat = 0);
    void add_item(std::int64_t id, py::object object);
    void add_descriptor(py::object descriptor);
};

heap_wrapper::heap_wrapper(std::int64_t heap_cnt, bug_compat_mask bug_compat)
    : heap(heap_cnt), bug_compat(bug_compat)
{
}

void heap_wrapper::add_item(std::int64_t id, py::object object)
{
    py::object buffer = object.attr("to_buffer")();
    item_buffers.emplace_back(buffer);
    const auto &view = item_buffers.back().view;
    heap::add_item(id, view.buf, view.len);
}

void heap_wrapper::add_descriptor(py::object object)
{
    heap::add_descriptor(py::extract<descriptor>(object.attr("to_raw")(bug_compat)));
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
        bug_compat_mask bug_compat,
        std::size_t max_packet_size,
        double rate,
        std::size_t max_heaps = Base::default_max_heaps,
        std::size_t buffer_size = Base::default_buffer_size)
        : Base(
            pool.get_io_service(),
            make_endpoint(pool.get_io_service(), hostname, port),
            heap_address_bits, bug_compat, max_packet_size, rate,
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

    class_<heap_wrapper, boost::noncopyable>("Heap", init<std::int64_t, bug_compat_mask>())
        .def("add_item", &heap_wrapper::add_item, with_custodian_and_ward<1, 3>())
        .def("add_descriptor", &heap_wrapper::add_descriptor);

    {
        typedef udp_stream_wrapper<stream_wrapper<udp_stream> > T;
        class_<T, boost::noncopyable>("UdpStream", init<
                thread_pool_wrapper &, std::string, int, int, bug_compat_mask,
                std::size_t, double, optional<std::size_t, std::size_t> >()[
                    with_custodian_and_ward<1, 2>()])
            .def("send_heap", &T::send_heap);
    }

    {
        typedef udp_stream_wrapper<asyncio_stream_wrapper<udp_stream> > T;
        class_<T, boost::noncopyable>("UdpStreamAsyncio", init<
                thread_pool_wrapper &, std::string, int, int, bug_compat_mask,
                std::size_t, double, optional<std::size_t> >()[
                    with_custodian_and_ward<1, 2>()])
            .add_property("fd", &T::get_fd)
            .def("async_send_heap", &T::async_send_heap)
            .def("flush", &T::flush)
            .def("process_callbacks", &T::process_callbacks);
    }
}

} // namespace send
} // namespace spead
