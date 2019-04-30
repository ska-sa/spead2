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
#include <boost/asio.hpp>
#include <boost/optional.hpp>
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
 * Wrapper for passing sockets from Python to C++. It extracts the file
 * descriptor and checks that it is of the correct type and protocol, and
 * provides a factory to create the Boost.Asio variant.
 */
template<typename SocketType>
class socket_wrapper
{
private:
    typename SocketType::protocol_type protocol;
    int fd;

public:
    socket_wrapper() : protocol(SocketType::protocol_type::v4()), fd(-1) {}
    socket_wrapper(typename SocketType::protocol_type protocol, int fd)
        : protocol(protocol), fd(fd) {}

    SocketType copy(boost::asio::io_service &io_service) const
    {
        int fd2 = ::dup(fd);
        if (fd2 == -1)
        {
            PyErr_SetFromErrno(PyExc_OSError);
            throw pybind11::error_already_set();
        }
        return SocketType(io_service, protocol, fd2);
    }
};

extern template class socket_wrapper<boost::asio::ip::udp::socket>;
extern template class socket_wrapper<boost::asio::ip::tcp::socket>;
extern template class socket_wrapper<boost::asio::ip::tcp::acceptor>;

boost::asio::ip::address make_address_no_release(
    boost::asio::io_service &io_service, const std::string &hostname,
    boost::asio::ip::resolver_query_base::flags flags);

/**
 * Issue a Python deprecation.
 *
 * Note that this might throw, due to the interface of PyErr_WarnEx.
 */
void deprecation_warning(const char *msg);

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

// Set up logging to go to the Python logging framework
void register_logging();
// Set up atexit handlers needed to ensure that exit_stopper works as advertised
void register_atexit();

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
            // Pragmas are to work around https://gcc.gnu.org/bugzilla/show_bug.cgi?id=86922
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
            return (obj.*Ptr)(std::forward<Args>(args)...);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
        }
    };

    template<Return (Class::*Ptr)(Args...) const>
    struct PTMFWrapperConst
    {
        typedef Return result_type;
        Return operator()(const T &obj, Args... args) const
        {
            // Pragmas are to work around https://gcc.gnu.org/bugzilla/show_bug.cgi?id=86922
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
            return (obj.*Ptr)(std::forward<Args>(args)...);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
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

namespace pybind11
{
namespace detail
{

template<typename SocketType>
struct type_caster<spead2::socket_wrapper<SocketType>>
{
public:
    PYBIND11_TYPE_CASTER(spead2::socket_wrapper<SocketType>, _("socket.socket"));

    bool load(handle src, bool)
    {
        int fd = -1;
        // Workaround for https://github.com/pybind/pybind11/issues/1473
        if (!hasattr(src, "fileno"))
            return false;
        try
        {
            fd = src.attr("fileno")().cast<int>();
        }
        catch (std::exception &)
        {
            return false;
        }

        // Determine whether this is IPv4 or IPv6
        sockaddr_storage addr;
        socklen_t addrlen = sizeof(addr);
        int ret = getsockname(fd, (sockaddr *) &addr, &addrlen);
        if (ret == -1)
            return false;
        if (addr.ss_family != AF_INET && addr.ss_family != AF_INET6)
            return false;
        auto protocol = (addr.ss_family == AF_INET) ?
            SocketType::protocol_type::v4() : SocketType::protocol_type::v6();

        // Check that the protocol (e.g. TCP or UDP) matches
        int type;
        socklen_t optlen = sizeof(type);
        ret = getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &optlen);
        if (ret == -1)
            return false;
        if (type != protocol.type())
            return false;

        value = spead2::socket_wrapper<SocketType>(protocol, fd);
        return true;
    }
};

/* Old versions of boost::optional don't implement emplace (required by
 * pybind11::optional_caster), so we implement the conversion manually.
 */
template<typename SocketType>
struct type_caster<boost::optional<spead2::socket_wrapper<SocketType>>>
{
    PYBIND11_TYPE_CASTER(boost::optional<spead2::socket_wrapper<SocketType>>, _("Optional[socket.socket]"));

    bool load(handle src, bool convert)
    {
        if (!src)
            return false;
        else if (src.is_none())
            return true;
        make_caster<spead2::socket_wrapper<SocketType>> inner_caster;
        if (!inner_caster.load(src, convert))
            return false;
        value = cast_op<spead2::socket_wrapper<SocketType> &&>(std::move(inner_caster));
        return true;
    }
};

} // namespace detail
} // namespace pybind11
#endif // SPEAD2_PY_COMMON_H
