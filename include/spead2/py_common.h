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
#include <atomic>
#include <list>
#include <stdexcept>
#include <type_traits>
#include <functional>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_memory_pool.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_logging.h>
#include <spead2/common_ringbuffer.h>
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
 * Helper to ensure that an asynchronous class is stopped when the module is
 * unloaded. A class that launches work asynchronously should contain one of
 * these as a member. It is initialised with a function object that stops the
 * work of the class. The stop function must in turn call @ref reset.
 */
class exit_stopper
{
private:
    std::list<std::function<void()>>::iterator entry;

public:
    explicit exit_stopper(std::function<void()> callback);

    /// Deregister the callback
    void reset();

    ~exit_stopper() { reset(); }
};

/**
 * Wrapper around @ref thread_pool that drops the GIL during blocking operations.
 */
class thread_pool_wrapper : public thread_pool
{
private:
    exit_stopper stopper{[this] { stop(); }};
public:
    /* Simply using thread_pool::thread_pool doesn't work because the default
     * constructor is deleted as stopper is not default-constructible, even
     * though it doesn't actually need to be.
     */
    template<typename ...Args>
    explicit thread_pool_wrapper(Args&&... args)
        : thread_pool(std::forward<Args>(args)...) {}

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

/**
 * Logger function object that passes log messages to Python. To avoid blocking
 * the caller while waiting for the GIL, it passes the log messages through a
 * ring buffer to a dedicated thread.
 */
class log_function_python
{
private:
    static constexpr unsigned int num_levels = 3;
    static const char *const level_methods[num_levels];

    exit_stopper stopper{[this] { stop(); }};
    pybind11::object log_methods[num_levels];
    std::atomic<bool> overflowed;
    ringbuffer<std::pair<log_level, std::string>> ring;
    std::thread thread;

    void run();

public:
    log_function_python() = default;
    explicit log_function_python(pybind11::object logger, std::size_t ring_size = 1024);

    ~log_function_python() { stop(); }

    // Directly log a message (GIL must be held)
    void log(log_level level, const std::string &msg) const;
    // Callback for the spead2 logging framework
    void operator()(log_level level, const std::string &msg);
    void stop();
};

// Like pybind11::buffer::request, but allows extra flags to be passed
pybind11::buffer_info request_buffer_info(pybind11::buffer &buffer, int extra_flags);

void register_module(pybind11::module m);

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
