#include <boost/test/unit_test.hpp>
#include <vector>
#include <memory>
#include "common_memory_pool.h"
#include "common_thread_pool.h"

namespace spead2
{
namespace unittest
{

BOOST_AUTO_TEST_SUITE(common)
BOOST_AUTO_TEST_SUITE(memory_pool)

// Repeatedly allocates memory from a memory pool to check that the refilling
// code does not crash.
BOOST_AUTO_TEST_CASE(memory_pool_refill)
{
    spead2::thread_pool tpool;
    std::shared_ptr<spead2::memory_pool> pool = std::make_shared<spead2::memory_pool>(
        tpool, 1024 * 1024, 2 * 1024 * 1024, 8, 4, 2);
    std::vector<spead2::memory_pool::pointer> pointers;
    for (int i = 0; i < 100; i++)
        pointers.push_back(pool->allocate(1024 * 1024));
}

BOOST_AUTO_TEST_SUITE_END()  // memory_pool
BOOST_AUTO_TEST_SUITE_END()  // common

}} // namespace spead2::unittest
