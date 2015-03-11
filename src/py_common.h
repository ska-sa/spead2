/**
 * @file
 */

#ifndef SPEAD_PY_COMMON_H
#define SPEAD_PY_COMMON_H

#include <memory>

#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <boost/version.hpp>
#include <cassert>
#include <mutex>
#include <stdexcept>
#include "common_ringbuffer.h"
#include "common_thread_pool.h"

namespace spead
{

class stop_iteration : public std::exception
{
public:
    using std::exception::exception;
};

/**
 * Wrapper for std::string that converts to a Python bytes object instead
 * of a str (Unicode) object.
 */
class bytestring : public std::string
{
public:
    using std::string::string;
};

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

    ~thread_pool_wrapper();
    void stop();
};

/**
 * Semaphore variant that releases the GIL during waits, and throws an
 * exception if interrupted by SIGINT in the Python process.
 */
class semaphore_gil : public semaphore
{
public:
    int get();
};

/* Older versions of boost don't understand std::shared_ptr properly. This is
 * in the spead namespace so that it will be found by ADL when considering
 * std::shared_ptr<spead::mem_pool>.
 */
#if BOOST_VERSION < 105300
template<typename T>
T *get_pointer(const std::shared_ptr<T> &p)
{
    return p.get();
}
#endif

} // namespace spead

#endif // SPEAD_PY_COMMON_H
