/* Copyright 2016 SKA South Africa
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

#ifndef SPEAD2_COMMON_MEMORY_ALLOCATOR_H
#define SPEAD2_COMMON_MEMORY_ALLOCATOR_H

#include <memory>
#include <cstdint>
#include <cstddef>
#include <utility>

namespace spead2
{

class memory_allocator;

/**
 * Pointer that implements the NullablePointer concept, but contains a
 * reference to the allocator that allocated it, and an arbitrary user
 * handle.
 */
template<typename T>
class memory_allocation
{
private:
    T *ptr;
    std::shared_ptr<memory_allocator> allocator;
    void *user;

public:
    memory_allocation(std::nullptr_t ptr = nullptr) : ptr(ptr), user(nullptr) {}
    memory_allocation(T *ptr, std::shared_ptr<memory_allocator> allocator, void *user = nullptr)
        : ptr(ptr), allocator(std::move(allocator)), user(user) {}
    explicit operator bool() { return ptr; }
    bool operator==(memory_allocation other) const { return ptr == other.ptr; }
    bool operator!=(memory_allocation other) const { return ptr != other.ptr; }
    T &operator*() const { return *ptr; }
    T *operator->() const { return ptr; }
    T *get() const { return ptr; }
    T &operator[](std::size_t i) { return ptr[i]; }

    std::shared_ptr<memory_allocator> get_allocator() const { return allocator; }
    void *get_user() const { return user; }
};

/**
 * Polymorphic class for managing memory allocations in a memory pool. This
 * can be overloaded to provide custom memory allocations.
 */
class memory_allocator : public std::enable_shared_from_this<memory_allocator>
{
public:
    class deleter
    {
    public:
        typedef memory_allocation<std::uint8_t> pointer;
        void operator()(pointer ptr);
    };

    typedef std::unique_ptr<std::uint8_t[], deleter> pointer;

    virtual ~memory_allocator() = default;

    /**
     * Allocate @a size bytes of memory. The default implementation uses @c new
     * and pre-faults the memory.
     *
     * @param size         Number of bytes to allocate
     * @param hint         Usage-dependent extra information
     * @returns Pointer to newly allocated memory
     * @throw std::bad_alloc if allocation failed
     */
    virtual pointer allocate(std::size_t size, void *hint);

protected:
    void prefault(std::uint8_t *ptr, std::size_t size);

private:
    /**
     * Free memory previously returned from @ref allocate.
     *
     * @param ptr          Value returned by @ref allocate
     * @param user         User-defined handle returned by @ref allocate
     */
    virtual void free(std::uint8_t *ptr, void *user);
};

/**
 * Allocator that uses mmap. This is useful for large allocations, where the
 * cost of going to the kernel (bypassing malloc) and the page-grained
 * padding is justified. The main reasons to use this over the default
 * allocator are:
 * - pointers are page-aligned, which is useful for things like direct I/O
 * - on Linux is uses @c MAP_POPULATE, which avoids lots of cache pollution
 *   when pre-faulting the hard way.
 * - it is possible to specify additional flags, e.g. MAP_HUGETLB or
 *   MAP_LOCKED.
 *
 * @internal
 *
 * The user data pointer is used to store the length of the mapping to pass
 * to munmap (via uintptr_t).
 */
class mmap_allocator : public memory_allocator
{
public:
    const int flags;

    explicit mmap_allocator(int flags = 0);
    virtual pointer allocate(std::size_t size, void *hint) override;

private:
    virtual void free(std::uint8_t *ptr, void *user) override;
};

} // namespace spead2

#endif // SPEAD2_COMMON_MEMORY_ALLOCATOR_H
