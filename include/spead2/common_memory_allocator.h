/* Copyright 2016, 2021 National Research Foundation (SARAO)
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
 * Polymorphic class for managing memory allocations in a memory pool. This
 * can be overloaded to provide custom memory allocations.
 */
class memory_allocator : public std::enable_shared_from_this<memory_allocator>
{
private:
    // Function object for backwards compatibility. It keeps a shared pointer
    // to the allocator alive and calls its free function.
    class legacy_deleter
    {
    private:
        struct state_t
        {
            std::shared_ptr<memory_allocator> allocator;
            void *user;

            state_t(std::shared_ptr<memory_allocator> &&allocator, void *user)
                : allocator(std::move(allocator)), user(user) {}
        };

        /* The state needs to be held behind an additional pointer indirection
         * to allow the call operator to be const and still reset the shared_ptr
         * to the allocator. And it can't be held by unique_ptr because
         * std::function is copyable. Potentially it could be held by raw
         * pointer, but as it is not expected to actually be copied there should
         * be relatively little overhead in using shared_ptr.
         */
        std::shared_ptr<state_t> state;
    public:
        legacy_deleter() = default;
        legacy_deleter(std::shared_ptr<memory_allocator> &&allocator, void *user);
        void operator()(std::uint8_t *ptr) const;  ///< Call the allocator to free the memory

        const std::shared_ptr<memory_allocator> &get_allocator() const { return state->allocator; }
        void *get_user() const { return state->user; }
    };

public:
    /**
     * Deleter for pointers allocated by this allocator.
     *
     * This class derives from @c std::function, so it can be constructed
     * with any deleter at run time. In particular, this means that
     * @c std::unique_ptr<std::uint8_t[], D> can be converted to
     * @c memory_allocator::pointer for any deleter @c D.
     *
     * For backwards compatibility, it can also be constructed from a shared
     * pointer to an allocator and an arbitrary void pointer, in which case the
     * deletion is performed by calling @ref memory_allocator::free.
     */
    class deleter : public std::function<void(std::uint8_t *)>
    {
    public:
        using std::function<void(std::uint8_t *)>::function;

        explicit deleter(std::shared_ptr<memory_allocator> allocator, void *user = nullptr)
            : std::function<void(std::uint8_t *)>(legacy_deleter(std::move(allocator), user)) {}

        /**
         * If the backwards-compatibility constructor was used, return the stored
         * allocator. Otherwise, returns a null pointer.
         */
        const std::shared_ptr<memory_allocator> &get_allocator() const;
        /**
         * If the backwards-compatibility constructor was used, return the stored
         * user pointer. Otherwise, returns a null pointer.
         */
        void *get_user() const;
    };

    typedef std::unique_ptr<std::uint8_t[], deleter> pointer;

    virtual ~memory_allocator() = default;

    /**
     * Allocate @a size bytes of memory. The default implementation uses @c new
     * and pre-faults the memory.
     *
     * The @c pointer type uses @ref memory_allocator::deleter as the deleter.
     * If the memory needs to be freed by something other than
     * <code>delete[]</code> then pass a function object when constructing the
     * pointer.
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
     * Free memory previously returned from @ref allocate. This is kept
     * only for backwards compatibility, so that existing allocators that
     * override it will continue to work.
     *
     * @param ptr          Value returned by @ref allocate
     * @param user         User-defined handle stored in the deleter by @ref allocate
     */
    virtual void free(std::uint8_t *ptr, void *user);
};

/**
 * Allocator that uses mmap. This is useful for large allocations, where the
 * cost of going to the kernel (bypassing malloc) and the page-grained
 * padding is justified. The main reasons to use this over the default
 * allocator are:
 * - pointers are page-aligned, which is useful for things like direct I/O
 * - on Linux it uses @c MAP_POPULATE, which avoids lots of cache pollution
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
    const int flags;         ///< Requested flags given to constructor
    const bool prefer_huge;  ///< Whether to prefer huge pages

    /**
     * Constructor.
     *
     * @param flags        Extra flags to pass on to mmap
     * @param prefer_huge  If true, allocations will try to use huge pages
     *                     (if supported by the OS), and fall back to normal
     *                     pages if that fails.
     */
    explicit mmap_allocator(int flags = 0, bool prefer_huge = false);

    virtual pointer allocate(std::size_t size, void *hint) override;
};

} // namespace spead2

#endif // SPEAD2_COMMON_MEMORY_ALLOCATOR_H
