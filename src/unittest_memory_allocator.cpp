#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <memory>
#include "common_memory_allocator.h"

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(common)
BOOST_AUTO_TEST_SUITE(memory_allocator)

typedef boost::mpl::list<spead2::memory_allocator, spead2::mmap_allocator> test_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(allocator_test, T, test_types)
{
    typedef typename T::pointer pointer;
    std::shared_ptr<T> allocator = std::make_shared<T>();
    pointer ptr = allocator->allocate(12345, nullptr);
    ptr.reset();
}

BOOST_AUTO_TEST_SUITE_END()  // memory_allocator
BOOST_AUTO_TEST_SUITE_END()  // common

}} // namespace spead2::unittest
