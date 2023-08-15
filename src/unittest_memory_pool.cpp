/* Copyright 2016, 2017, 2019, 2021, 2023 National Research Foundation (SARAO)
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
 *
 * Unit tests for common_memory_pool.
 */

#include <boost/test/unit_test.hpp>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <thread>
#include <chrono>
#include <spead2/common_memory_pool.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_logging.h>

namespace spead2::unittest
{

BOOST_AUTO_TEST_SUITE(common)
BOOST_AUTO_TEST_SUITE(memory_pool)

BOOST_AUTO_TEST_CASE(default_constructible)
{
    spead2::memory_pool pool;
}

// Repeatedly allocates memory from a memory pool to check that the refilling
// code does not crash.
BOOST_AUTO_TEST_CASE(memory_pool_refill)
{
    spead2::thread_pool tpool;
    std::shared_ptr<spead2::memory_pool> pool = std::make_shared<spead2::memory_pool>(
        tpool, 1024 * 1024, 2 * 1024 * 1024, 8, 4, 2);
    pool->set_warn_on_empty(false);
    std::vector<spead2::memory_pool::pointer> pointers;
    for (int i = 0; i < 100; i++)
        pointers.push_back(pool->allocate(1024 * 1024, nullptr));
    // Give the refiller time to refill completely
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

class mock_allocator : public spead2::memory_allocator
{
public:
    struct record
    {
        bool allocate;
        std::size_t size;
        void *ptr;
    };

    struct deleter
    {
        std::shared_ptr<mock_allocator> allocator;
        // ptr + 1, used to check that state is preserved with each pointer
        std::uint8_t *next_ptr;

        deleter(std::shared_ptr<mock_allocator> allocator, std::uint8_t *ptr)
            : allocator(std::move(allocator)), next_ptr(ptr + 1)
        {
        }

        void operator()(std::uint8_t *ptr) const
        {
            BOOST_CHECK_EQUAL(ptr + 1, next_ptr);
            allocator->records.push_back(record{false, 0, ptr});
            allocator->saved.emplace_back(ptr);
        }
    };

    std::vector<record> records;
    std::vector<std::unique_ptr<std::uint8_t[]>> saved;

    virtual pointer allocate(std::size_t size, void *hint) override
    {
        (void) hint;
        std::uint8_t *ptr = new std::uint8_t[size];
        records.push_back(record{true, size, ptr});
        std::shared_ptr<mock_allocator> shared_this =
            std::static_pointer_cast<mock_allocator>(shared_from_this());
        return pointer(ptr, deleter(shared_this, ptr));
    }
};

// Overrides the logger for a test to record all log messages
class logger_fixture
{
private:
    std::function<void(spead2::log_level, const std::string &)> old_logger;

public:
    std::map<log_level, std::vector<std::string>> messages;

    logger_fixture()
    {
        auto logger = [this] (spead2::log_level level, const std::string &msg)
        {
            messages[level].push_back(msg);
        };
        old_logger = spead2::set_log_function(logger);
    }

    ~logger_fixture()
    {
        spead2::set_log_function(old_logger);
    }
};

// Check that the underlying deleter is preserved.
BOOST_AUTO_TEST_CASE(memory_pool_pass_deleter)
{
    typedef spead2::memory_allocator::pointer pointer;
    std::shared_ptr<mock_allocator> allocator = std::make_shared<mock_allocator>();
    std::shared_ptr<spead2::memory_pool> pool = std::make_shared<spead2::memory_pool>(
        1024, 2048, 2, 1, allocator);
    pool->set_warn_on_empty(false);

    // Pooled allocation, comes from the pre-allocated slot
    pointer p1 = pool->allocate(1536, nullptr);
    mock_allocator::deleter *base_deleter;
    base_deleter = spead2::memory_pool::get_base_deleter(p1).target<mock_allocator::deleter>();
    BOOST_TEST_REQUIRE(base_deleter != nullptr);
    BOOST_TEST(base_deleter->next_ptr == p1.get() + 1);
    // Pooled allocations, newly allocated
    pointer p2 = pool->allocate(1500, nullptr);
    pointer p3 = pool->allocate(1024, nullptr);

    // Free all the pointers. p3 should be dropped
    p1.reset();
    p2.reset();
    p3.reset();

    // Make another allocation, which should come from the pool (p1)
    pointer p4 = pool->allocate(1600, nullptr);
    // Allocation not from the pool
    pointer p5 = pool->allocate(2049, nullptr);
    base_deleter = spead2::memory_pool::get_base_deleter(p5).target<mock_allocator::deleter>();
    BOOST_TEST_REQUIRE(base_deleter != nullptr);
    BOOST_TEST(base_deleter->next_ptr == p5.get() + 1);

    // Release our reference to the pool. It should be destroyed once p4 is freed
    pool.reset();
    // Free remaining pointers
    p4.reset();
    p5.reset();

    // Check results
    BOOST_REQUIRE_EQUAL(allocator->records.size(), 8);
    BOOST_CHECK_EQUAL(allocator->records[0].allocate, true);
    BOOST_CHECK_EQUAL(allocator->records[0].size, 2048);
    BOOST_CHECK_EQUAL(allocator->records[1].allocate, true);
    BOOST_CHECK_EQUAL(allocator->records[1].size, 2048);
    BOOST_CHECK_EQUAL(allocator->records[2].allocate, true);
    BOOST_CHECK_EQUAL(allocator->records[2].size, 2048);
    BOOST_CHECK_EQUAL(allocator->records[3].allocate, false);
    BOOST_CHECK_EQUAL(allocator->records[3].ptr, allocator->records[2].ptr);
    BOOST_CHECK_EQUAL(allocator->records[4].allocate, true);
    BOOST_CHECK_EQUAL(allocator->records[4].size, 2049);
    BOOST_CHECK_EQUAL(allocator->records[5].allocate, false);
    BOOST_CHECK_EQUAL(allocator->records[5].ptr, allocator->records[0].ptr);
    BOOST_CHECK_EQUAL(allocator->records[6].allocate, false);
    BOOST_CHECK_EQUAL(allocator->records[6].ptr, allocator->records[1].ptr);
    BOOST_CHECK_EQUAL(allocator->records[7].allocate, false);
    BOOST_CHECK_EQUAL(allocator->records[7].ptr, allocator->records[4].ptr);
}

// Check that a warning is issued when the memory pool becomes empty, if and
// only if the warning is enabled.
BOOST_FIXTURE_TEST_CASE(memory_pool_warn_on_empty, logger_fixture)
{
    auto pool = std::make_shared<spead2::memory_pool>(1024, 2048, 1, 1);
    BOOST_CHECK_EQUAL(pool->get_warn_on_empty(), true);
    auto ptr1 = pool->allocate(1024, nullptr);
    // Wasn't empty, should be no messages
    BOOST_CHECK_EQUAL(messages[spead2::log_level::warning].size(), 0);
    auto ptr2 = pool->allocate(1024, nullptr);
    // Now was empty, should be a warning
    BOOST_CHECK_EQUAL(messages[spead2::log_level::warning].size(), 1);

    pool->set_warn_on_empty(false);
    BOOST_CHECK_EQUAL(pool->get_warn_on_empty(), false);
    auto ptr3 = pool->allocate(1024, nullptr);
    // Empty again, but should be no new warning
    BOOST_CHECK_EQUAL(messages[spead2::log_level::warning].size(), 1);
}

BOOST_AUTO_TEST_SUITE_END()  // memory_pool
BOOST_AUTO_TEST_SUITE_END()  // common

} // namespace spead2::unittest
