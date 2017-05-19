/* Copyright 2015, 2017 SKA South Africa
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

#ifndef SPEAD2_PY_COMMON_H
#define SPEAD2_PY_COMMON_H

#include <memory>
#include <utility>
#include <boost/system/system_error.hpp>
#include <cassert>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_memory_pool.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace spead2
{

/// Wrapper for generating Python IOError from a boost error code
class boost_io_error : public boost::system::system_error
{
public:
    using boost::system::system_error::system_error;
};

/**
 * Wraps a std::string member of a class to be read as pybind11::bytes i.e.
 * disable UTF-8 conversion.
 */
template<typename T>
static inline pybind11::cpp_function bytes_getter(std::string T::*ptr)
{
    return pybind11::cpp_function(
        [ptr](const T &obj) { return pybind11::bytes(obj.*ptr); });
}

template<typename T>
static inline pybind11::cpp_function bytes_setter(std::string T::*ptr)
{
    return pybind11::cpp_function(
        [ptr](T &obj, const pybind11::bytes &value) { obj.*ptr = value; });
}

/**
 * Wrapper around @ref thread_pool that drops the GIL during blocking operations.
 */
class thread_pool_wrapper : public thread_pool
{
public:
    using thread_pool::thread_pool;

    ~thread_pool_wrapper();
    void stop();
};

/**
 * Semaphore variant that releases the GIL during waits, and throws an
 * exception if interrupted by SIGINT in the Python process.
 */
template<typename Semaphore>
class semaphore_gil : public Semaphore
{
public:
    using Semaphore::Semaphore;
    int get();
};

template<typename Semaphore>
int semaphore_gil<Semaphore>::get()
{
    int result;
    {
        pybind11::gil_scoped_release gil;
        result = Semaphore::get();
    }
    if (result == -1)
    {
        // Allow SIGINT to abort the wait
        if (PyErr_CheckSignals() == -1)
            throw pybind11::error_already_set();
    }
    return result;
}

class log_function_python
{
private:
    pybind11::object logger;
public:
    log_function_python() = default;
    explicit log_function_python(pybind11::object logger) : logger(std::move(logger)) {}

    void operator()(log_level level, const std::string &msg)
    {
        pybind11::gil_scoped_acquire gil;

        static const char *const level_methods[] =
        {
            "warning",
            "info",
            "debug"
        };
        unsigned int level_idx = static_cast<unsigned int>(level);
        assert(level_idx < sizeof(level_methods) / sizeof(level_methods[0]));
        logger.attr(level_methods[level_idx])("%s", msg);
    }
};

// Like pybind11::buffer::request, but allows extra flags to be passed
pybind11::buffer_info request_buffer_info(pybind11::buffer &buffer, int extra_flags);

pybind11::module register_module();

namespace detail
{

// Convert a base class pointer to member function to the derived class
template<typename Derived, typename Base, typename Result, typename ...Args>
static inline auto up(Result (Base::*func)(Args...)) -> Result (Derived::*)(Args...) 
{
    return static_cast<Result (Derived::*)(Args...)>(func);
}

template<typename Derived, typename Base, typename Result, typename ...Args>
static inline auto up(Result (Base::*func)(Args...) const) -> Result (Derived::*)(Args...) const
{
    return static_cast<Result (Derived::*)(Args...) const>(func);
}

template<typename Derived, typename Base, typename T>
static inline auto up(T Base::*ptr) -> T Derived::*
{
    return static_cast<T Derived::*>(ptr);
}

// Fallback when argument is not a pointer to member
template<typename T, typename U>
static inline U &&up(U &&u) { return std::forward<U>(u); }

} // namespace detail

/**
 * Wrapper around py::class_ that works around
 * https://github.com/pybind/pybind11/issues/854.
 *
 * Note: the reimplemented methods must be called *first* in the chain,
 * because any methods that aren't will return the base class.
 */
template<typename T, typename ...ClassExtra>
class class_ : public pybind11::class_<T, ClassExtra...>
{
private:
    typedef pybind11::class_<T, ClassExtra...> base;

public:
    using pybind11::class_<T, ClassExtra...>::class_;

    // Wraps member function to change the return to *this.
#define SPEAD2_GENERIC_WRAP(func)                           \
    template<typename ...Args> class_ &func(Args&& ...args) \
    {                                                       \
        base::func(std::forward<Args>(args)...);            \
        return *this;                                       \
    }

    SPEAD2_GENERIC_WRAP(def)
    SPEAD2_GENERIC_WRAP(def_readonly)
    SPEAD2_GENERIC_WRAP(def_readwrite)
    SPEAD2_GENERIC_WRAP(def_property)
    SPEAD2_GENERIC_WRAP(def_property_readonly)
    SPEAD2_GENERIC_WRAP(def_property_static)
    SPEAD2_GENERIC_WRAP(def_property_readonly_static)
    SPEAD2_GENERIC_WRAP(def_buffer)
#undef SPEAD2_GENERIC_WRAP

    template<typename Func, typename ...Extra>
    class_ &def(const char *name, Func &&func, Extra&& ...extra)
    {
        base::def(name, detail::up<T>(std::forward<Func>(func)), std::forward<Extra>(extra)...);
        return *this;
    }

    template<typename D, typename ...Extra>
    class_ &def_readonly(const char *name, D &&pm, Extra&& ...extra)
    {
        base::def_readonly(name, detail::up<T>(std::forward<D>(pm)), std::forward<Extra>(extra)...);
        return *this;
    }

    template<typename D, typename ...Extra>
    class_ &def_readwrite(const char *name, D &&pm, Extra&& ...extra)
    {
        base::def_readwrite(name, detail::up<T>(std::forward<D>(pm)), std::forward<Extra>(extra)...);
        return *this;
    }

    template<typename Getter, typename ...Extra>
    class_ &def_property_readonly(const char *name, Getter &&fget, Extra&&... extra)
    {
        base::def_property_readonly(name, detail::up<T>(std::forward<Getter>(fget)),
                                    std::forward<Extra>(extra)...);
        return *this;
    }

    template<typename Getter, typename Setter, typename ...Extra>
    class_ &def_property(const char *name, Getter &&fget, Setter &&fset, Extra&& ...extra)
    {
        base::def_property(name,
                           detail::up<T>(std::forward<Getter>(fget)),
                           detail::up<T>(std::forward<Setter>(fset)),
                           std::forward<Extra>(extra)...);
        return *this;
    }
};

namespace detail
{

/* Some magic for defining the SPEAD2_PTMF macro, which wraps a pointer to
 * member function into a stateless class which avoids using a run-time
 * function pointer.
 *
 * The type T and the Class template parameter are separated because if T
 * derives from Class and a member function foo is defined in Class, then the
 * type of &T::foo is actually <code>Return (Class::*)(Args...)</code> rather
 * than <code>Return (T::*)(Args...)</code>.
 */
template<typename T, typename Return, typename Class, typename... Args>
struct PTMFWrapperGen
{
    template<Return (Class::*Ptr)(Args...)>
    struct PTMFWrapper
    {
        typedef Return result_type;
        Return operator()(T &obj, Args... args) const
        {
            return (obj.*Ptr)(std::forward<Args>(args)...);
        }
    };

    template<Return (Class::*Ptr)(Args...) const>
    struct PTMFWrapperConst
    {
        typedef Return result_type;
        Return operator()(const T &obj, Args... args) const
        {
            return (obj.*Ptr)(std::forward<Args>(args)...);
        }
    };

    template<Return (Class::*Ptr)(Args...)>
    static constexpr PTMFWrapper<Ptr> make_wrapper() noexcept { return PTMFWrapper<Ptr>(); }

    template<Return (Class::*Ptr)(Args...) const>
    static constexpr PTMFWrapperConst<Ptr> make_wrapper() noexcept { return PTMFWrapperConst<Ptr>(); }
};

// This function is never defined, and is only used as a helper for decltype
template<typename T, typename Return, typename Class, typename... Args>
PTMFWrapperGen<T, Return, Class, Args...> ptmf_wrapper_type(Return (Class::*ptmf)(Args...));
template<typename T, typename Return, typename Class, typename... Args>
PTMFWrapperGen<T, Return, Class, Args...> ptmf_wrapper_type(Return (Class::*ptmf)(Args...) const);

#define SPEAD2_PTMF(Class, Func) \
    (decltype(::spead2::detail::ptmf_wrapper_type<Class>(&Class::Func))::template make_wrapper<&Class::Func>())

} // namespace detail

} // namespace spead2

#endif // SPEAD2_PY_COMMON_H
