/**
 * @file
 */

#ifndef SPEAD_PY_COMMON_H
#define SPEAD_PY_COMMON_H

#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <cassert>
#include <mutex>
#include "common_ringbuffer.h"

namespace spead
{

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

    ~buffer_view()
    {
        PyBuffer_Release(&view);
    }
};

/**
 * Ringbuffer variant that releases the GIL while waiting for data, and aborts
 * if there was a @c KeyboardInterrupt.
 */
template<typename T>
class ringbuffer_wrapper : public ringbuffer<T>
{
public:
    using ringbuffer<T>::ringbuffer;

    T pop();
};

template<typename T>
T ringbuffer_wrapper<T>::pop()
{
    release_gil gil;
    std::unique_lock<std::mutex> lock(this->mutex);
    while (this->empty_unlocked())
    {
        this->data_cond.wait_for(lock, std::chrono::seconds(1));
        // Allow interpreter to catch KeyboardInterrupt. The timeout
        // ensures that we wake up periodically to check for this,
        // even if the signal doesn't cause spurious wakeup.
        gil.acquire();
        if (PyErr_CheckSignals() == -1)
            boost::python::throw_error_already_set();
        gil.release();
    }
    return this->pop_unlocked();
}

} // namespace spead

#endif // SPEAD_PY_COMMON_H
