/**
 * @file
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL spead2_ARRAY_API
#define NO_IMPORT_ARRAY
#include <boost/python.hpp>
#include <stdexcept>
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
private:
    static boost::asio::ip::udp::endpoint make_endpoint(
        boost::asio::io_service &io_service, const std::string &hostname, int port);

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

class udp_stream_wrapper : public stream_wrapper<udp_stream>
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
        std::size_t max_heaps = DEFAULT_MAX_HEAPS)
        : stream_wrapper<udp_stream>(
            pool.get_io_service(),
            make_endpoint(pool.get_io_service(), hostname, port),
            heap_address_bits, bug_compat, max_packet_size, rate, max_heaps)
    {
    }
};

boost::asio::ip::udp::endpoint udp_stream_wrapper::make_endpoint(
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

    class_<udp_stream_wrapper, boost::noncopyable>("UdpStream", init<
            thread_pool_wrapper &, std::string, int, int, bug_compat_mask,
            std::size_t, double, optional<std::size_t> >()[
                with_custodian_and_ward<1, 2>()])
        .def("send_heap", &udp_stream_wrapper::send_heap);
}

} // namespace send
} // namespace spead
