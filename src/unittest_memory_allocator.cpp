/* Copyright 2016, 2019, 2021, 2023 National Research Foundation (SARAO)
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
 * Unit tests for common_memory_allocator.
 */

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <vector>
#include <memory>
#include <utility>
#include <spead2/common_memory_allocator.h>

namespace spead2::unittest
{

BOOST_AUTO_TEST_SUITE(common)
BOOST_AUTO_TEST_SUITE(memory_allocator)

// Wrapper to pass prefer_huge to the mmap_allocator constructor
class huge_mmap_allocator : public spead2::mmap_allocator
{
public:
    huge_mmap_allocator() : spead2::mmap_allocator(0, true) {}
};

class legacy_allocator : public spead2::memory_allocator
{
public:
    std::vector<std::pair<std::uint8_t *, void *>> deleted;

    virtual pointer allocate(std::size_t size, void *hint) override
    {
        /* The size is copied to a volatile to outsmart the compiler, which
         * sees that there is code that tries to allocate more than PTRDIFF_MAX
         * elements (to test out-of-memory) and complains.
         */
        volatile std::size_t size_copy = size;
        std::uint8_t *data = new std::uint8_t[size_copy];
        return pointer(data, deleter(shared_from_this(), hint));
    };

private:
    virtual void free(std::uint8_t *ptr, void *user) override
    {
        delete[] ptr;
        deleted.emplace_back(ptr, user);
    }
};

typedef boost::mpl::list<
    spead2::memory_allocator,
    spead2::mmap_allocator,
    huge_mmap_allocator,
    legacy_allocator> test_types;

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

BOOST_AUTO_TEST_CASE_TEMPLATE(out_of_memory, T, test_types)
{
    std::shared_ptr<T> allocator = std::make_shared<T>();
    BOOST_CHECK_THROW(allocator->allocate(SIZE_MAX - 1, nullptr), std::bad_alloc);
}

BOOST_AUTO_TEST_CASE(legacy)
{
    auto allocator = std::make_shared<legacy_allocator>();
    auto ptr = allocator->allocate(123, nullptr);
    BOOST_TEST(allocator->deleted.empty());
    std::uint8_t *raw = ptr.get();
    ptr.reset();
    BOOST_TEST(allocator->deleted.size() == 1);
    BOOST_TEST(allocator->deleted[0].first == raw);
    BOOST_TEST(allocator->deleted[0].second == nullptr);

    // Now a pointer with a non-null user value (set via hint)
    ptr = allocator->allocate(234, &raw);
    raw = ptr.get();
    ptr.reset();
    BOOST_TEST(allocator->deleted.size() == 2);
    BOOST_TEST(allocator->deleted[1].first == raw);
    BOOST_TEST(allocator->deleted[1].second == &raw);
}

BOOST_AUTO_TEST_SUITE_END()  // memory_allocator
BOOST_AUTO_TEST_SUITE_END()  // common

} // namespace spead2::unittest
