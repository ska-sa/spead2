#include <array>
#include <utility>
#include <boost/test/unit_test.hpp>
#include <spead2/send_streambuf.h>

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(send)
BOOST_AUTO_TEST_SUITE(streambuf)

namespace
{

/* streambuf that does not implement overflow(). It will thus report EOF as
 * soon as it runs out of its pre-defined buffer space.
 */
class array_streambuf : public std::streambuf
{
public:
    array_streambuf(char *first, char *last)
    {
        setp(first, last);
    }
};

}   // anonymous namespace

/* Send data to a streambuf that returns an EOF error
 * (the success case is tested more thoroughly via Python).
 */
BOOST_AUTO_TEST_CASE(send_fail)
{
    spead2::thread_pool tp;
    spead2::send::heap h;
    h.add_item(0x1234, 0x5678);
    std::array<char, 20> buffer;
    array_streambuf sb(buffer.begin(), buffer.end());
    spead2::send::streambuf_stream stream(tp, sb);
    std::promise<std::pair<boost::system::error_code, std::size_t>> result_promise;
    auto handler = [&](const boost::system::error_code &ec, std::size_t bytes_transferred)
    {
        result_promise.set_value(std::make_pair(ec, bytes_transferred));
    };
    stream.async_send_heap(h, handler);
    auto result = result_promise.get_future().get();
    BOOST_CHECK_EQUAL(result.first, boost::asio::error::eof);
    BOOST_CHECK_EQUAL(result.second, 0);
}

BOOST_AUTO_TEST_SUITE_END()  // streambuf
BOOST_AUTO_TEST_SUITE_END()  // send

}} // namespace spead2::unittest
