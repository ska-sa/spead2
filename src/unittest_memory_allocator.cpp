#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <memory>
#include <spead2/common_memory_allocator.h>

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(common)
BOOST_AUTO_TEST_SUITE(memory_allocator)

// Wrapper to pass prefer_huge to the mmap_allocator constructor
class huge_mmap_allocator : public spead2::mmap_allocator
{
public:
    huge_mmap_allocator() : spead2::mmap_allocator(0, true) {}
};

typedef boost::mpl::list<spead2::memory_allocator, spead2::mmap_allocator, huge_mmap_allocator> test_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(allocator_test, T, test_types)
{
    typedef typename T::pointer pointer;
    std::shared_ptr<T> allocator = std::make_shared<T>();
    pointer ptr = allocator->allocate(12345, nullptr);
    // Scribble on the memory to make sure it is really allocated
    for (std::size_t i = 0; i < 12345; i++)
        ptr[i] = 1;
    ptr.reset();
}

BOOST_AUTO_TEST_SUITE_END()  // memory_allocator
BOOST_AUTO_TEST_SUITE_END()  // common

}} // namespace spead2::unittest
