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
 *
 * Backport of @c std::make_unique_for_overwrite from C++20.
 */

#ifndef SPEAD2_COMMON_UNIQUE_H
#define SPEAD2_COMMON_UNIQUE_H

#include <memory>
#include <type_traits>

namespace spead2::detail
{

template<typename T>
std::enable_if_t<!std::is_array_v<T>, std::unique_ptr<T>> make_unique_for_overwrite()
{
    return std::unique_ptr<T>(new T);
}

template<typename T>
std::enable_if_t<std::is_array_v<T> && std::extent_v<T> == 0, std::unique_ptr<T>> make_unique_for_overwrite(std::size_t n)
{
    return std::unique_ptr<T>(new std::remove_extent_t<T>[n]);
}

template<typename T, typename... Args>
std::enable_if_t<std::extent_v<T> != 0, std::unique_ptr<T>> make_unique_for_overwrite(Args&&...) = delete;

} // namespace spead2::detail

#endif // SPEAD2_COMMON_UNIQUE_H
