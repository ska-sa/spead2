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

#ifndef SPEAD2_PY_COMMON_H
#define SPEAD2_PY_COMMON_H

#include <memory>
#include <utility>
#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <boost/version.hpp>
#include <boost/system/system_error.hpp>
#include <cassert>
#include <mutex>
#include <stdexcept>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_memory_pool.h>
#include <spead2/common_thread_pool.h>
#include <spead2/common_logging.h>

namespace spead2
{

class stop_iteration : public std::exception
{
public:
    using std::exception::exception;
};

/// Wrapper for generating Python IOError from a boost error code
class boost_io_error : public boost::system::system_error
{
public:
    using boost::system::system_error::system_error;
};

/**
 * Wrapper for std::string that converts to a Python bytes object instead
 * of a str (Unicode) object.
 */
class bytestring : public std::string
{
public:
    using std::string::string;

    bytestring(const std::string &s)
        : std::string(s)
    {
    }

    bytestring(std::string &&s)
        : std::string(std::move(s))
    {
    }
};

/**
 * Function object utility to provide getter and setter access to a std::string
 * element of a structure as if it were declared as a bytestring. Used by
 * make_bytestring_getter and make_bytestring_setter.
 */
template<class Data, class Class>
class bytestring_member
{
public:
    explicit bytestring_member(Data Class::*ptr) : ptr(ptr) {}

    bytestring operator()(Class &c) const
    {
        return c.*ptr;
    }

    void operator()(Class &c, bytestring value) const
    {
        c.*ptr = std::move(value);
    }

private:
    Data Class::*ptr;
};

template<class Data, class Class>
boost::python::object make_bytestring_getter(Data Class::*ptr)
{
    using namespace boost::python;
    return make_function(
        bytestring_member<Data, Class>(ptr),
        return_value_policy<return_by_value>(),
        boost::mpl::vector2<bytestring, Class &>());
}

template<class Data, class Class>
boost::python::object make_bytestring_setter(Data Class::*ptr)
{
    using namespace boost::python;
    return make_function(
        bytestring_member<Data, Class>(ptr),
        default_call_policies(),
        boost::mpl::vector3<void, Class &, bytestring>());
}

/**
 * RAII wrapper that releases the Python Global Interpreter Lock on
 * construction and reacquires it on destruction. It is also possible to
 * freely acquire and release it during the lifetime; if it is released on
 * destruction, it is reacquired.
 *
 * One thread must @em not have two instances of this object.
 */
class release_gil
{
private:
    PyThreadState *save = nullptr;

public:
    release_gil()
    {
        release();
    }

    ~release_gil()
    {
        if (save != nullptr)
            PyEval_RestoreThread(save);
    }

    void release()
    {
        assert(save == nullptr);
        save = PyEval_SaveThread();
    }

    void acquire()
    {
        assert(save != nullptr);
        PyEval_RestoreThread(save);
        save = nullptr;
    }
};

/**
 * RAII class to acquire the GIL in a non-Python thread.
 */
class acquire_gil
{
private:
    PyGILState_STATE gstate;
public:
    acquire_gil()
    {
        gstate = PyGILState_Ensure();
    }

    ~acquire_gil()
    {
        PyGILState_Release(gstate);
    }
};

/**
 * Wraps access to a Python buffer-protocol object. On construction, it
 * fetches the buffer, and on destruction it releases it. At present, only
 * @c PyBUF_SIMPLE is supported, but it could easily be extended.
 */
class buffer_view : public boost::noncopyable
{
public:
    Py_buffer view;

    explicit buffer_view(boost::python::object obj)
    {
        if (PyObject_GetBuffer(obj.ptr(), &view, PyBUF_SIMPLE) != 0)
            boost::python::throw_error_already_set();
    }

    buffer_view()
    {
        view.ndim = -1;  // mark as invalid to prevent release
    }

    // Allow moving
    buffer_view(buffer_view &&other) noexcept : view(other.view)
    {
        other.view.ndim = -1; // mark as invalid
    }

    buffer_view &operator=(buffer_view &&other)
    {
        if (view.ndim >= 0)
        {
            acquire_gil gil;
            PyBuffer_Release(&view);
        }
        view = other.view;
        other.view.ndim = -1;
        return *this;
    }

    ~buffer_view()
    {
        if (view.ndim >= 0)
        {
            acquire_gil gil;
            PyBuffer_Release(&view);
        }
    }
};

/**
 * Wrapper around @ref thread_pool that drops the GIL during blocking operations.
 */
class thread_pool_wrapper : public thread_pool
{
public:
    using thread_pool::thread_pool;
    thread_pool_wrapper(int num_threads, boost::python::list affinity);

    ~thread_pool_wrapper();
    void stop();
};

/**
 * Container for a thread pool handle. It's put into a separate class so that
 * it can be inherited from by wrapper classes that need it, earlier in the
 * inheritance chain than objects that depend on it. That's necessary to obtain
 * the correct destructor ordering.
 */
struct thread_pool_handle_wrapper
{
    boost::python::handle<> thread_pool_handle;
};

/**
 * Wrapper around @ref memory_pool that holds a Python handle to the
 * underlying thread pool, if any.
 */
class memory_pool_wrapper : public thread_pool_handle_wrapper, public memory_pool
{
public:
    using memory_pool::memory_pool;
};

/// Like @ref thread_pool_handle_wrapper, but for a memory allocator
struct memory_allocator_handle_wrapper
{
    boost::python::handle<> memory_allocator_handle;
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
    release_gil gil;
    int result = Semaphore::get();
    if (result == -1)
    {
        // Allow SIGINT to abort the wait
        gil.acquire();
        if (PyErr_CheckSignals() == -1)
            boost::python::throw_error_already_set();
    }
    return result;
}

/* Older versions of boost don't understand std::shared_ptr properly. This is
 * in the spead2 namespace so that it will be found by ADL when considering
 * std::shared_ptr<spead2::memory_pool_wrapper>.
 *
 * Due to https://svn.boost.org/trac/boost/ticket/7473, Boost does not detect
 * standard library support in Clang 3.4.
 */
#if BOOST_VERSION < 105300 || defined(BOOST_NO_CXX11_SMART_PTR)
template<typename T>
T *get_pointer(const std::shared_ptr<T> &p)
{
    return p.get();
}
#endif

/**
 * Used instead of with_custodian_and_ward_postcall to ensure that one object
 * holds a reference to another. This seems to be more reliable than
 * with_custodian_and_ward, which in some cases seems to not respect
 * dependencies when the interpreter is shut down.
 */
template<typename T, typename P, boost::python::handle<> P::*handle_ptr,
    std::size_t custodian, std::size_t ward,
    class BasePolicy_ = boost::python::default_call_policies>
struct store_handle_postcall : BasePolicy_
{
    static_assert(custodian != ward, "object must not hold reference to itself");

    static PyObject* postcall(PyObject *args, PyObject *result)
    {
        std::size_t arity = PyTuple_GET_SIZE(args);
        if (custodian > arity || ward > arity)
        {
            PyErr_SetString(PyExc_IndexError,
                            "store_handle_postcall: argument index out of range");
            return nullptr;
        }

        result = BasePolicy_::postcall(args, result);
        if (result == nullptr)
            return nullptr;

        PyObject *owner = custodian > 0 ? PyTuple_GET_ITEM(args, custodian - 1) : result;
        PyObject *child = ward > 0 ? PyTuple_GET_ITEM(args, ward - 1) : result;
        boost::python::extract<T &> extractor(owner);
        try
        {
            T &target = extractor();
            target.*handle_ptr = boost::python::handle<>(boost::python::borrowed(child));
        }
        catch (boost::python::error_already_set)
        {
            Py_XDECREF(result);
            return nullptr;
        }
        return result;
    }
};

class log_function_python
{
private:
    boost::python::object logger;
public:
    typedef void result_type;

    log_function_python() = default;
    explicit log_function_python(const boost::python::object &logger) : logger(logger) {}

    void operator()(log_level level, const std::string &msg)
    {
        acquire_gil gil;

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

void register_module();

} // namespace spead2

#endif // SPEAD2_PY_COMMON_H
