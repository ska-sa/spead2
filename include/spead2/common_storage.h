/* Copyright 2023 National Research Foundation (SARAO)
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

#ifndef SPEAD2_COMMON_STORAGE_H
#define SPEAD2_COMMON_STORAGE_H

#include <cstdint>
#include <new>
#include <utility>

namespace spead2::detail
{

/**
 * Similar to @c std::aligned_storage, but with a safer API. It has the same
 * size and alignment as T, but is initially uninitialised. The contained object
 * can be constructed and destroyed as needed.
 *
 * The user must ensure the object is destroyed when the storage is.
 */
template<typename T>
class storage
{
private:
    alignas(T) std::uint8_t raw[sizeof(T)];

public:
    T *get() { return std::launder(reinterpret_cast<T *>(&raw)); }
    const T *get() const { return std::launder(reinterpret_cast<const T *>(&raw)); }
    T *operator->() { return get(); }
    const T *operator->() const { return get(); }
    T &operator *() { return *get(); }
    const T &operator *() const { return *get(); }

    template<typename... Args>
    T *construct(Args&&... args)
    {
        return new(&raw) T(std::forward<Args>(args)...);
    }

    void destroy() { get()->~T(); }
};

} // namespace spead2::detail

#endif // SPEAD2_COMMON_STORAGE_H
