Python bindings
===============

The Python bindings are implemented using pybind11_, which handles most of the
work of exposing C++ classes and members to Python.

.. _pybind11: https://pybind11.readthedocs.io/

Global Interpreter Lock
-----------------------
The Python Global Interpreter Lock (GIL) is both good news and bad news for
the bindings. The good news is that it ensures that (by default) only one
Python thread can be calling the spead2 API at a time, which provides some
basic thread safety. Here's the bad news (but also read the
:external+pybind11:ref:`pybind11 documentation <gil>` on the topic):

- Without additional steps, blocking calls to the spead2 API would prevent
  other Python threads from making progress. Mostly this just hurts
  performance, but it can even lead to deadlocks. This is mostly handled by
  judicious use of :cpp:class:`!pybind11::gil_scoped_release` in functions
  that wait for events or take C++ locks.

- When a thread does not hold the GIL (either because it released it, or
  because it is a C++ thread created internally) it cannot safely call Python
  APIs. While :external+pybind11:cpp:class:`pybind11::object <object>` is a
  very convenient interface for reference counting, it can cause accidental
  use of the reference counting APIs through destructors or copy constructors.
  For any class that embeds a Python object, it is important
  to know from which threads it could be destroyed (and in some cases the core
  spead2 design has had to be modified to account for this). Some pieces of
  code that do not expect to do any reference counting choose to use
  :external+pybind11:cpp:class:`pybind11::handle <handle>` instead to
  eliminate the risk.

- If a worker thread tries to acquire the GIL, this could block for a
  significant period of time, which in turn could cause high latency for
  time-critical work (such as processing packets).

  Logging to the Python logging system is one area where this used to be
  problematic. That has been solved by putting log messages (as C++ strings)
  on a ringbuffer and using a dedicated worker thread to pass the log messages
  to Python. If this worker thread is blocked, log messages simply accumulate
  in the ringbuffer, and eventually log messages get dropped, but the thread
  that is doing the logging is not blocked.

  A similar issue is notifying Python code that some asynchronous work has
  completed. This is covered in a later section.

Note that once :pep:`703` is implemented, the spead2 bindings may need
significant rework to support no-GIL mode.

Semaphores
----------

.. cpp:namespace-push:: spead2

spead2 uses semaphores extensively for producer/consumer relationships between
threads. Dropping the GIL before calling a blocking function will partially
help, but has two potential issues:

1. If the blocking function waits on a semaphore and then takes action that
   uses the Python API, it will need to re-acquire the GIL *before* taking
   that action.

2. If the user presses Ctrl-C, it will not interrupt the program: it will
   interrupt the low-level semaphore wait (with ``EINTR``), but the default
   semaphore implementation in spead2 will simply retry that wait. Graceful
   handling of Ctrl-C requires using :c:func:`PyErr_CheckSignals` to give
   Python a chance to handle the interrupt.

To address both concerns, several functions (particularly in
:cpp:class:`ringbuffer`) take a variadic number of extra arguments, which are
passed on to :cpp:func:`semaphore_get`. The Python bindings provide an
overload of :cpp:func:`semaphore_get`, selected by passing an instance of the
empty class :cpp:class:`gil_release_tag`, that handles both releasing the GIL
and checking :c:func:`PyErr_CheckSignals` each time the semaphore wait is
interrupted.

Asynchronous callbacks
----------------------
The C++ interface for sending heaps asynchronously takes a callback which is
executed when the heap has been transmitted. While the high-level Python
interface exposed to users wraps this into an :py:class:`asyncio.Future`, the
low-level interface exposed from C++ to Python still takes a callback.
However, the callback is Python code and can only be called with the GIL held.
Furthermore, since :mod:`asyncio` is not thread-safe, it should be run in the
thread that's running the asyncio event loop, rather than the C++ thread
that's doing the networking.

To solve these problems, the C++-level callback doesn't directly invoke the
Python callback. Instead, it puts it in a vector of deferred callbacks. To
wake up the event loop without locking the GIL, it writes data to a file
descriptor (eventfd in Linux, but it's abstracted by
:cpp:class:`semaphore_fd`), which the asyncio event loop watches. The
event loop thread then requests the stream to execute all its deferred
callbacks.

As an optimisation, the file descriptor is only touched if the callback list
was empty. Thus, a rapid burst of complete heaps only requires one wakeup.
This requires some careful use of a mutex to correctly handle
the case where new callbacks are added while the callback list is being
processed.

.. _interpreter-shutdown:

Interpreter shutdown
--------------------
If spead2 objects are still present when the Python interpreter is shut down,
their destructors may try to interact with the Python API after it is too late
to safely do so. For example, the logging system may try to interact with the
GIL after the GIL has already been destroyed. Thus, some objects need to be
shut down earlier in the interpreter shutdown process, and this is achieved
using the :mod:`atexit` module. An :cpp:class:`exit_stopper` class simplifies
the process.
