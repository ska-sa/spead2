/* Copyright 2015 SKA South Africa
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

#ifndef SPEAD2_COMMON_BIND_H
#define SPEAD2_COMMON_BIND_H

#include <tuple>
#include <utility>
#include <cstddef>
#include <type_traits>

namespace spead2
{
namespace detail
{

/**
 * Crude re-implementation of std::index_sequence from C++14.
 */
template<std::size_t... N>
class index_sequence
{
public:
    static constexpr std::size_t size() { return sizeof...(N); }
};

template<typename T, std::size_t N>
class index_sequence_append;

/// Given an index sequence type, returns the type with N appended
template<std::size_t... I, std::size_t N>
class index_sequence_append<index_sequence<I...>, N>
{
public:
    typedef index_sequence<I..., N> type;
};

template<std::size_t N>
class make_index_sequence_impl : public index_sequence_append<
        typename make_index_sequence_impl<N - 1>::type,
        N - 1>
{
};

template<>
class make_index_sequence_impl<0>
{
public:
    typedef index_sequence<> type;
};

/**
 * Crude re-implementation of std::make_index_sequence from C++14.
 */
template<std::size_t N>
using make_index_sequence = make_index_sequence_impl<N>;

/**
 * Implementation class for @ref reference_bind. The template types here are
 * either const lvalue references, lvalue references, or rvalue references.
 */
template<typename T, typename... Args>
class reference_binder
{
public:
    typedef typename std::result_of<T(Args...)>::type result_type;

private:
    T function;
    std::tuple<Args...> args;
    static constexpr bool is_noexcept = noexcept(std::declval<T>()(std::declval<Args>()...));

    template<std::size_t... I>
    result_type invoke(index_sequence<I...>) noexcept(is_noexcept)
    {
        return std::forward<T>(function)(
            std::forward<typename std::tuple_element<I, decltype(args)>::type>(
                std::get<I>(args))...);
    }

public:
    reference_binder(T &&function, Args&&... args)
        : function(std::forward<T>(function)),
        args(std::forward<Args>(args)...)
    {
    }

    result_type operator()() noexcept(is_noexcept)
    {
        // The index sequence is used to index the tuple elements
        return invoke(typename make_index_sequence<sizeof...(Args)>::type());
    }
};

/**
 * A variant of std::bind that does perfect forwarding. Unlike std::bind,
 * reference arguments remain references, and rvalue references can also be
 * passed through. It also does not support placeholders, reference_wrapper,
 * sub-bind expressions, or pointer-to-member functions. Note that temporaries
 * passed are kept as rvalue references and their lifetime is @em not extended.
 * They must be used by the end of the full expression. Similarly, the function
 * itself is forwarded.
 *
 * Note that because temporaries are forwarded as rvalue reference, the
 * resulting function object has a non-const operator(), and it can only safely
 * be called once.
 */
template<typename T, typename... Args>
reference_binder<T&&, Args&&...> reference_bind(T &&function, Args&&... args)
{
    return reference_binder<T&&, Args&&...>(std::forward<T>(function), std::forward<Args>(args)...);
}

} // namespace detail
} // namespace spead2

#endif // SPEAD2_COMMON_BIND_H
